#!/usr/bin/env python
# coding: utf-8
# Author: Jianiang Ren

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.io import load_obj

from core.models.modules import DINOBase, StyleUNetLight
from core.models.modules.omg_transformer import TransformerDecoderWithProjection
from core.libs.utils_renderer import (
    render_gaussian,
    project_points_to_normalized_pixel_coordinates,
    get_normlized_z_buffer,
    build_camera_matrices,
    remove_faces_with_vertices,
)
from core.libs.utils_perceptual import FacePerceptualLoss
from core.libs.flame_model import FLAMEModel


# Vertex slice ranges for each subdivision level: (start, end) per head segment.
_HEAD_SPLITS = {
    0: [(0, 5023)],
    1: [(0, 5023), (5023, 20018)],
    2: [(0, 5023), (5023, 20018), (20018, 79936)],
}


class OMGAvatar(nn.Module):
    def __init__(self, model_cfg=None, **kwargs):
        super().__init__()

        self.base_dino_model = DINOBase(output_dim=256)
        for param in self.base_dino_model.dino_model.parameters():
            param.requires_grad = False

        dim = model_cfg.DIM
        self.transformer = TransformerDecoderWithProjection(
                        d_model=dim,
                        nhead=8,
                        num_layers=model_cfg.N_TRANSFORMER_LAYERS,
                        memory_dim=768,
                        input_dim = dim  ,
                        dim_feedforward=1024,
                        dropout=0.1
                        )

        # dir_encoder
        n_harmonic_dir = 4
        self.direnc_dim = n_harmonic_dir * 2 * 3 + 3 # 27
        self.harmo_encoder = HarmonicEmbedding(n_harmonic_dir)
        self.subdivide_times = model_cfg.SUBDIVISION
        self.head_splits = _HEAD_SPLITS[self.subdivide_times]
        self.head_total_points = self.head_splits[-1][1]

        self.head_base = nn.Parameter(torch.randn(5023, dim), requires_grad=True)
        self.gs_head_generators = nn.ModuleList(HeadGSGenerator(in_dim=dim) for i in range(self.subdivide_times+1))
        self.gs_generator_offset = OffsetGEnerator(in_dim=256)
        self.gs_shoulder_generator = ConvGSGenerator(in_dim=256, dir_dim=self.direnc_dim)

        self.cam_params = {'focal_x': 12.0, 'focal_y': 12.0, 'size': [512, 512]}
        self.upsampler = StyleUNetLight(in_size=512, in_dim=32, out_dim=3, out_size=512) # neural renderer
        self.percep_loss = FacePerceptualLoss(loss_type='l1', weighted=True)

        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, #data_cfg.FLAME_SCALE,
                                      no_lmks=True)
        for param in self.flame_model.parameters():
            param.requires_grad = False

        flame_mask = np.load('./assets/FLAME_mask/FLAME_masks.pkl', allow_pickle=True, encoding='latin1')
        neck = flame_mask['neck']
        boundary = flame_mask['boundary']
        neck_vert_index = neck.tolist() + boundary.tolist()

        flame_template_mesh_path = './assets/human_parametric_models/flame_assets/flame/head_template_mesh.obj'
        verts, faces, aux = load_obj(flame_template_mesh_path, load_textures=False)
        self._verts = verts.unsqueeze(0)
        faces = faces.verts_idx.unsqueeze(0)

        self._meshes = Meshes(verts=self._verts, faces=faces)
        self.flame_faces = faces
        self.faces_without_neck = remove_faces_with_vertices(faces, neck_vert_index)

        self.subdivider = SubdivideMeshes(self._meshes)
        self.add_shoulder = True

    def subdivide_mesh(self, in_mesh, subdivide_times, feats=None):
        '''
        in_mesh: [bs,vertices,3]
        '''
        if feats is None:
            batch_size = in_mesh.shape[0]
            if subdivide_times > 0:
                meshes = Meshes(verts=in_mesh, faces=self.flame_faces.tile(batch_size, 1, 1).to(in_mesh.device))
                for _ in range(subdivide_times):
                    subdivided_meshes = self.subdivider(meshes, feats=None)
                    meshes = subdivided_meshes
                out_mesh_pts = torch.stack(subdivided_meshes.verts_list())
                return out_mesh_pts, subdivided_meshes
            else:
                meshes = Meshes(verts=in_mesh, faces=self.flame_faces.tile(batch_size, 1, 1).to(in_mesh.device))
                return in_mesh, meshes
        else:
            batch_size = in_mesh.shape[0]
            if subdivide_times > 0:
                meshes = Meshes(verts=in_mesh, faces=self.flame_faces.tile(batch_size, 1, 1).to(in_mesh.device))
                for _ in range(subdivide_times):
                    subdivided_meshes, subdivided_feats = self.subdivider(meshes, feats)
                    meshes = subdivided_meshes
                    feats = subdivided_feats
                out_mesh_pts = torch.stack(subdivided_meshes.verts_list())
                return out_mesh_pts, subdivided_meshes, subdivided_feats
            else:
                meshes = Meshes(verts=in_mesh, faces=self.flame_faces.tile(batch_size, 1, 1).to(in_mesh.device))
                return in_mesh, meshes, feats

    def recon_avatar(self, batch):
        batch_size = batch['f_image'].shape[0]
        t_image, t_bbox = batch['t_image'], batch['t_bbox']
        t_transform = batch['t_transform']
        t_shape, t_pose, t_exp, t_eye = batch['t_shape'], batch['t_pose'], batch['t_exp'], batch['t_eye']
        f_image, f_planes, f_transform = batch['f_image'], batch['f_planes'], batch['f_transform']
        f_exp = batch['f_exp']
        f_pose = batch['f_pose']
        f_eye = batch['f_eye']
        f_shape = batch['f_shape']

        output_size = int(math.sqrt(f_planes['plane_points'].shape[1])) # 296
        f_feature1, _, f_feature_multiscale = self.base_dino_model(f_image) #  (bs, 256, 296, 296)  (bs, 768) (bs,1369,  768)

        f_image_512 = torchvision.transforms.functional.resize(f_image, (512, 512), antialias=True)
        f_head_mask = (torch.sum(f_image_512, dim=1, keepdim=True) > 1e-3).float()

        plane_direnc = self.harmo_encoder(f_planes['plane_dirs']) # (bs,27)

        _head_base = self.head_base[None]
        _head_base = _head_base.expand(batch_size, -1, -1)  # (bs, 5023, 256)

        feature_volume_5023 = self.transformer(vert_feature=_head_base, image_feature=f_feature_multiscale)
        self.v_offset = self.gs_generator_offset(feature_volume_5023, plane_direnc)

        f_points = self.flame_model(  # shape can be f_shape or t_shape
            shape_params=f_shape, pose_params=f_pose,
            expression_params=f_exp, eye_pose_params=f_eye, verts_offset=None # v_offset
        ).float()

        f_sub_points, f_sub_meshs, feature_volume_sub = self.subdivide_mesh(
            f_points, self.subdivide_times, feats=feature_volume_5023
        )

        t_points = self.flame_model(  # shape can be f_shape or t_shape
            shape_params=f_shape,
            pose_params=t_pose,
            expression_params=t_exp,
            eye_pose_params=t_eye,
            verts_offset=self.v_offset
        ).float()

        t_sub_points, t_sub_meshs = self.subdivide_mesh(t_points, self.subdivide_times, feats=None)

        n_coords, ndc = project_points_to_normalized_pixel_coordinates(f_sub_points, f_transform, self.cam_params)
        coords = (n_coords * f_feature1.shape[-1]).clamp(0, f_feature1.shape[-1]-1) # (bs, 2000, 2)

        grid = torch.zeros_like(coords)
        grid[..., 0] = coords[..., 0] / (f_feature1.shape[-1] - 1) * 2 - 1  # Normalize x to [-1, 1]
        grid[..., 1] = coords[..., 1] / (f_feature1.shape[-2] - 1) * 2 - 1  # Normalize y to [-1, 1]
        grid = grid.unsqueeze(2)  # (bs, 2000, 1, 2)

        f_feature_sampling = F.grid_sample(f_feature1, grid, mode='bilinear', align_corners=True) # (bs,256, V, 1)
        f_feature_sampling = f_feature_sampling[...,0].permute(0, 2, 1)  # (bs, V, 256)

        with torch.no_grad():
            f_z_buffer, occluded_vertices_mask = get_normlized_z_buffer(f_transform, self.cam_params, f_sub_meshs, ndc)

            t_face_meshes = Meshes(verts=t_points, faces=self.faces_without_neck.tile(batch_size, 1, 1).to(t_points.device))
            t_face_z_buffer, _ = get_normlized_z_buffer(t_transform, self.cam_params, t_face_meshes, ndc)

            f_face_meshes = Meshes(verts=f_points, faces=self.faces_without_neck.tile(batch_size, 1, 1).to(t_points.device))
            f_face_z_buffer, _ = get_normlized_z_buffer(f_transform, self.cam_params, f_face_meshes, ndc)

        f_cloth_mask = (f_face_z_buffer > 0.95).float()
        f_cloth_mask[:,:,:int(f_cloth_mask.shape[2]* 3//4),:] = 0
        self.f_cloth_mask = f_cloth_mask

        z_buffer_mask = (f_z_buffer < 0.5).float()
        bg_mask = torch.clamp(f_head_mask*0 + 1 - z_buffer_mask, 0, 1) # (bs, 1, h, w)
        self.bg_mask = bg_mask

        cloth_mask_4dino = torchvision.transforms.functional.resize(bg_mask, (296, 296), antialias=True)
        cloth_mask_4dino = cloth_mask_4dino[:,0,:,:].unsqueeze(-1)

        occ_mask = occluded_vertices_mask.unsqueeze(-1).expand_as(f_feature_sampling).float()
        f_feature_combined = feature_volume_sub + f_feature_sampling * (1 - occ_mask)

        gs_params_subs = [
            self.gs_head_generators[i](f_feature_combined[:, s:e, :], plane_direnc)
            for i, (s, e) in enumerate(self.head_splits)
        ]
        gs_params_g = {
            k: torch.cat([p[k] for p in gs_params_subs], dim=1) for k in gs_params_subs[0].keys()
        }

        gs_params_g['xyz'] = t_sub_points

        gs_params_l0 = self.gs_shoulder_generator(f_feature1, plane_direnc)
        gs_params_l0['opacities'] = gs_params_l0['opacities'].reshape(batch_size, output_size, output_size, 1) * cloth_mask_4dino
        gs_params_l0['opacities'] = gs_params_l0['opacities'].reshape(batch_size, -1, 1)
        gs_params_l0['xyz'] = f_planes['plane_points'] + gs_params_l0['positions'] * f_planes['plane_dirs'][:, None]

        if batch_size == 1:
            valid_mask = gs_params_l0['opacities'] > 1e-4   # (bs, 87616, 1)
            valid_mask = valid_mask.squeeze()
            for k in gs_params_l0.keys():
                gs_params_l0[k] = gs_params_l0[k][0][valid_mask].unsqueeze(0)

        self.shoulder_points = gs_params_l0['xyz']

        if self.add_shoulder:
            gs_params = {k: torch.cat([gs_params_l0[k], gs_params_g[k]], dim=1) for k in gs_params_g.keys()}
        else:
            gs_params = gs_params_g

        self._gs_params = gs_params

        self.recon_out = {
            'gs_params': gs_params,
            'f_z_buffer': f_z_buffer,
            't_face_z_buffer': t_face_z_buffer,
            'f_face_z_buffer': f_face_z_buffer,
            'f_cloth_mask': self.f_cloth_mask,
            'bg_mask': self.bg_mask,
            'f_image_512': f_image_512,
            'occluded_vertices_mask': occluded_vertices_mask,
            't_sub_points': t_sub_points,
            't_points': t_points,
        }

    def set_subdivide_times(self, subdivide_times):
        self.subdivide_times = subdivide_times
        self.head_splits = _HEAD_SPLITS[self.subdivide_times]
        self.head_total_points = self.head_splits[-1][1]


    def forward(self, batch, is_training=True, enhance_shoulder=False):
        if is_training:
            self._gs_params = None

        batch_size = batch['f_image'].shape[0]
        t_image, t_bbox = batch['t_image'], batch['t_bbox']
        t_transform = batch['t_transform']
        t_shape, t_pose, t_exp, t_eye = batch['t_shape'], batch['t_pose'], batch['t_exp'], batch['t_eye']
        f_image, f_planes, f_transform = batch['f_image'], batch['f_planes'], batch['f_transform']
        f_exp = batch['f_exp']
        f_pose = batch['f_pose']
        f_eye = batch['f_eye']
        f_shape = batch['f_shape']

        if self._gs_params is None:
            self.recon_avatar(batch)
            t_sub_points = self.recon_out['t_sub_points']
        else:
            t_points = self.flame_model(  # shape can be f_shape or t_shape
                shape_params=f_shape, pose_params=t_pose,
                expression_params=t_exp, eye_pose_params=t_eye, verts_offset=self.v_offset
            ).float()
            t_sub_points, t_sub_meshs = self.subdivide_mesh(t_points, self.subdivide_times, feats=None)
 
        start_time = time.time()
        gs_params = self._gs_params

        if not enhance_shoulder:
            gs_params['xyz'][:, -self.head_total_points:] = t_sub_points

            render_out = render_gaussian(
                gs_params=gs_params, cam_matrix=t_transform, cam_params=self.cam_params
            )
        else:
            if is_training:
                gs_params_copy = self._gs_params.copy()
                focal_x, focal_y = self.cam_params['focal_x'], self.cam_params['focal_y']
                f_view_mat, _, _ = build_camera_matrices(f_transform, focal_x, focal_y)  # [8,4,4]
                t_view_mat, _, _ = build_camera_matrices(t_transform, focal_x, focal_y)  # [8,4,4]

                _shoulder_points = torch.cat([self.shoulder_points, torch.ones_like(self.shoulder_points[:, :,:1])], dim=-1)
                _t_head_points = torch.cat([t_sub_points, torch.ones_like(t_sub_points[:, :,:1])], dim=-1)
                _combined_points = torch.concat([_shoulder_points, _t_head_points], dim=1)

                num_points = _combined_points.shape[1]
                inter_M = t_view_mat.unsqueeze(1).expand(-1, num_points, -1, -1)

                _combined_points = _combined_points @ f_view_mat
                result = torch.einsum('bni,bnij->bnj', _combined_points, torch.inverse(inter_M))
                result = result[:,:,:3] / result[:,:,3:4]
                gs_params_copy['xyz'] = result

                # pose similar to src, expression similar to tgt
                render_only_exp = render_gaussian(gs_params=gs_params_copy, cam_matrix=t_transform, cam_params=self.cam_params)

                gs_params['xyz'][:, -self.head_total_points:] = t_sub_points

                render_out = render_gaussian(
                    gs_params=gs_params, cam_matrix=t_transform, cam_params=self.cam_params
                )
                gen_images_only_exp = render_only_exp['images']
            else:
                gs_params_copy = self._gs_params.copy()
                focal_x, focal_y = self.cam_params['focal_x'], self.cam_params['focal_y']
                f_view_mat, _, _ = build_camera_matrices(f_transform, focal_x, focal_y)  # [8,4,4]

                t_transform[:,:,-1] = f_transform[:,:,-1]  # use source camera translation
                t_view_mat, _, _ = build_camera_matrices(t_transform, focal_x, focal_y)  # [8,4,4]

                _shoulder_points = torch.cat([self.shoulder_points, torch.ones_like(self.shoulder_points[:, :,:1])], dim=-1)
                _t_head_points = torch.cat([t_sub_points, torch.ones_like(t_sub_points[:, :,:1])], dim=-1)
                _combined_points = torch.concat([_shoulder_points, _t_head_points], dim=1)

                num_points = _combined_points.shape[1]
                inter_M = t_view_mat.unsqueeze(1).expand(-1, num_points, -1, -1)

                _combined_points = _combined_points @ f_view_mat
                result = torch.einsum('bni,bnij->bnj', _combined_points, torch.inverse(inter_M))
                result = result[:,:,:3] / result[:,:,3:4]
                gs_params_copy['xyz'] = result

                render_only_exp = render_gaussian(gs_params=gs_params_copy, cam_matrix=t_transform, cam_params=self.cam_params)

                # enhance shoulder
                _combined_points = torch.concat([_shoulder_points, _t_head_points], dim=1)
                top_y = _t_head_points[0, 3461, 2]
                bottom_y = torch.min(_combined_points[0,:,2])
                alpha = torch.clamp((_combined_points[0,:,1] - bottom_y) / (top_y - bottom_y), 0, 1)  # bottom=0, top=1
                alpha = 1. - alpha
                alpha = torch.clip(2 * alpha - 0.8, 0, 1)

                f_view_mat_expand = f_view_mat.expand(alpha.shape[0], 4, 4)
                t_view_mat_expand = t_view_mat.expand(alpha.shape[0], 4, 4)

                if not enhance_shoulder:
                    alpha = alpha * 0

                inter_M = f_view_mat_expand * (1 - alpha.view(-1, 1, 1)) + t_view_mat_expand * alpha.view(-1, 1, 1)
                inter_M = inter_M.view(batch_size, alpha.shape[0], 4, 4)

                _combined_points = _combined_points @ f_view_mat
                result = torch.einsum('bni, bnij->bnj', _combined_points, torch.inverse(inter_M))
                result = result[:,:,:3] / result[:,:,3:4]
                gs_params['xyz'] = result

                render_out = render_gaussian(
                    gs_params=gs_params, cam_matrix=t_transform, cam_params=self.cam_params
                )
                gen_images_only_exp = render_only_exp['images']

        gen_images = render_out['images']

        if is_training:
            n_coords, _ = project_points_to_normalized_pixel_coordinates(t_sub_points, t_transform, self.cam_params)
            coords = (n_coords * 512).long().clamp(0, 512-1)  # [bs, 5023, 2]

            coord_vis_g = torch.zeros_like(gen_images[:, :1,:,:]).to(t_image.device)
            coord_vis_r = torch.zeros_like(gen_images[:, :1,:,:]).to(t_image.device)
            x_coords = coords[:, :, 0].long()  # (bs, 2000)
            y_coords = coords[:, :, 1].long()  # (bs, 2000)

            occluded_vertices_mask = self.recon_out['occluded_vertices_mask']
            visiable_x_coords = torch.where(occluded_vertices_mask, 0, x_coords)
            visiable_y_coords = torch.where(occluded_vertices_mask, 0, y_coords)
            batch_indices = torch.arange(gen_images.shape[0]).view(gen_images.shape[0], 1).expand(-1, x_coords.shape[-1])  # (bs, 2000)
            channel_indices = torch.zeros_like(x_coords)

            coord_vis_r[batch_indices, channel_indices, visiable_y_coords, visiable_x_coords] = 1
            coord_vis_g[batch_indices, channel_indices, y_coords, x_coords] = 1

            t_coord_vis = torch.cat([coord_vis_g, coord_vis_r, coord_vis_r], dim=1)

            pred_mask = (torch.sum(gen_images[:, :3,:,:], dim=1, keepdim=True) > 1e-3).float()
            pred_mask = pred_mask.repeat(1, 3, 1, 1)
            gt_mask = (torch.sum(t_image, dim=1, keepdim=True) > 1e-3).float()
            gt_mask = gt_mask.repeat(1, 3, 1, 1)

        sr_gen_images = self.upsampler(gen_images)
        if is_training:
            sr_gen_images_only_exp = self.upsampler(gen_images_only_exp)

        torch.cuda.synchronize()
        end_time = time.time()

        f_image_512 = self.recon_out['f_image_512']
        f_z_buffer = self.recon_out['f_z_buffer']
        f_face_z_buffer = self.recon_out['f_face_z_buffer']
        t_face_z_buffer = self.recon_out['t_face_z_buffer']
        f_cloth_mask = self.recon_out['f_cloth_mask']
        bg_mask = self.recon_out['bg_mask']
        t_points = self.recon_out['t_points']

        if is_training:
            results = {
                'f_image': f_image_512,
                't_image': t_image,
                't_bbox': t_bbox,
                't_points': t_points,
                'gen_image': gen_images[:, :3,:,:],
                'gen_images_only_exp': gen_images_only_exp[:, :3,:,:],
                'sr_gen_image': sr_gen_images,
                'sr_gen_images_only_exp': sr_gen_images_only_exp,
                'f_cloth_mask': f_cloth_mask.repeat(1, 3, 1, 1),
                'f_z_buffer': f_z_buffer.repeat(1, 3, 1, 1),
                'f_face_z_buffer': f_face_z_buffer.repeat(1, 3, 1, 1),
                't_face_z_buffer': t_face_z_buffer.repeat(1, 3, 1, 1),
                'pred_mask': pred_mask,
                'gt_mask': gt_mask,
                'bg_mask': bg_mask.repeat(1, 3, 1, 1),
                't_coord_vis': t_coord_vis,
                'v_offset': self.v_offset,
                '_gs_features': gen_images,
            }
        else:
            results = {
                'f_image': f_image_512,
                't_image': t_image,
                'gen_image': gen_images[:, :3,:,:],
                'sr_gen_image': sr_gen_images,
                'fps': 1. / (end_time - start_time),
            }

        return results
 
    def dump_intermediate_results(self, results, dump_dir, dump_name):
        if '_gs_features' in results:
            print(f'Dumping intermediate results to {dump_dir}/{dump_name}.npz')
            tgt = results['t_image']
            input_features = results['_gs_features']
            np.savez(f'{dump_dir}/{dump_name}.npz', input_features=input_features.detach().cpu().numpy(), tgt=tgt.detach().cpu().numpy())

    def calc_box_loss(self, image, gt_image, bbox, loss_fn, resize_size=512):
        def _resize(frames, tgt_size):
            frames = nn.functional.interpolate(
                frames, size=(tgt_size, tgt_size), mode='bilinear', align_corners=False, antialias=True
            )
            return frames
        bbox = bbox.clamp(min=0, max=1)
        bbox = (bbox * image.shape[-1]).long()
        pred_croped, gt_croped = [], []
        for idx, box in enumerate(bbox):
            gt_croped.append(_resize(gt_image[idx:idx+1, :, box[1]:box[3], box[0]:box[2]], resize_size))
            pred_croped.append(_resize(image[idx:idx+1, :, box[1]:box[3], box[0]:box[2]], resize_size))
        gt_croped = torch.cat(gt_croped, dim=0)
        pred_croped = torch.cat(pred_croped, dim=0)
        box_fn_loss = loss_fn(pred_croped, gt_croped)
        box_perc_loss = self.percep_loss(pred_croped, gt_croped) * 1e-2
        return box_fn_loss, box_perc_loss


class OffsetGEnerator(nn.Module):
    def __init__(self, in_dim=256, dir_dim=27, exp_dim=100, **kwargs):
        super().__init__()
        layer_in_dim = in_dim + dir_dim  # + exp_dim
        self.offset_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 3, bias=True),
            nn.Tanh()
        )

    def forward(self, input_features, plane_direnc):
        plane_direnc = plane_direnc[:, None].expand(-1, input_features.shape[1], -1)
        input_features = torch.cat([input_features, plane_direnc], dim=-1)
        offsets = self.offset_layers(input_features) * 0.05
        return offsets


class HeadGSGenerator(nn.Module):
    def __init__(self, in_dim=256, dir_dim=27, **kwargs):
        super().__init__()
        layer_in_dim = in_dim + dir_dim

        self.color_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 32, bias=True),
        )
        self.opacity_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True),
        )
        self.scale_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 3, bias=True)
        )
        self.rotation_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 4, bias=True),
        )
        self.offset_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 3, bias=True),
            nn.Tanh()
        )

    def forward(self, input_features, plane_direnc):
        plane_direnc = plane_direnc[:, None].expand(-1, input_features.shape[1], -1)
        input_features = torch.cat([input_features, plane_direnc], dim=-1)
        # color
        colors = self.color_layers(input_features)
        colors[..., :3] = torch.sigmoid(colors[..., :3])
        # opacity
        opacities = self.opacity_layers(input_features)
        opacities = torch.sigmoid(opacities)
        # scale
        scales = self.scale_layers(input_features)
        scales = torch.sigmoid(scales) * 0.05
        # rotation
        rotations = self.rotation_layers(input_features)
        rotations = nn.functional.normalize(rotations)

        return {'colors': colors, 'opacities': opacities, 'scales': scales, 'rotations': rotations}


class ConvGSGenerator(nn.Module):
    def __init__(self, in_dim=256, dir_dim=27, **kwargs):
        super().__init__()
        out_dim = 32 + 1 + 3 + 4 + 1 # color + opacity + scale + rotation + position
        self.gaussian_conv = nn.Sequential(
            nn.Conv2d(in_dim+dir_dim, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input_features, plane_direnc):
        plane_direnc = plane_direnc[:, :, None, None].expand(-1, -1, input_features.shape[2], input_features.shape[3]) # (bs,27,296,296)
        input_features = torch.cat([input_features, plane_direnc], dim=1)
        gaussian_params = self.gaussian_conv(input_features)
        # color
        colors = gaussian_params[:, :32]
        colors[..., :3] = torch.sigmoid(colors[..., :3])
        # opacity
        opacities = gaussian_params[:, 32:33]
        opacities = torch.sigmoid(opacities)
        # scale
        scales = gaussian_params[:, 33:36]
        scales = torch.sigmoid(scales) * 0.05
        # rotation
        rotations = gaussian_params[:, 36:40]
        rotations = nn.functional.normalize(rotations)
        # position
        positions = gaussian_params[:, 40:41]
        positions = torch.sigmoid(positions)
        results = {'colors': colors, 'opacities': opacities, 'scales': scales, 'rotations': rotations, 'positions': positions}
        for key in results.keys():
            results[key] = results[key].permute(0, 2, 3, 1).reshape(results[key].shape[0], -1, results[key].shape[1])
        return results


def expand_bbox(bbox, scale=1.1):
    xmin, ymin, xmax, ymax = bbox.unbind(dim=-1)
    cenx, ceny = (xmin + xmax) / 2, (ymin + ymax) / 2
    extend_size = torch.sqrt((ymax - ymin) * (xmax - xmin)) * scale
    extend_size = torch.min(extend_size, cenx*2)
    extend_size = torch.min(extend_size, ceny*2)
    extend_size = torch.min(extend_size, (1-cenx)*2)
    extend_size = torch.min(extend_size, (1-ceny)*2)
    xmine, xmaxe = cenx - extend_size / 2, cenx + extend_size / 2
    ymine, ymaxe = ceny - extend_size / 2, ceny + extend_size / 2
    return torch.stack([xmine, ymine, xmaxe, ymaxe], dim=-1)