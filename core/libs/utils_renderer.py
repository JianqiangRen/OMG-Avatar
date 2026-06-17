#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Modifications Copyright (c) 2025 Jianiang Ren

import math

import torch
from diff_gaussian_rasterization_32d import GaussianRasterizationSettings, GaussianRasterizer
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
)


NUM_CHANNELS = 32


def project_points_to_normalized_pixel_coordinates(points, cam_matrix, cam_params):
    """
    Projects points from world space to normalized pixel coordinates.

    Args:
        points: [batch_size, num_points, 3]
        cam_matrix: [batch_size, 3, 4]
        cam_params: dict with focal_x, focal_y, size

    Returns:
        normlized_coords: [batch_size, num_points, 2]
        ndc: [batch_size, num_points, 3]
    """
    focal_x, focal_y = cam_params['focal_x'], cam_params['focal_y']
    _, proj_mat, _ = build_camera_matrices(cam_matrix, focal_x, focal_y)
    # Convert to homogeneous coordinates
    _points = torch.cat([points, torch.ones_like(points[:, :, :1])], dim=-1)
    clip_space_coord = _points @ proj_mat
    ndc = clip_space_coord[:, :, :3] / clip_space_coord[:, :, 3:4]

    # Convert from NDC to normalized pixel coordinates in [0, 1]
    normlized_coords = ndc[:, :, :2] * 0.5 + 0.5
    normlized_coords = normlized_coords.clamp(min=0, max=1)
    return normlized_coords, ndc


def render_depth_map(cam_matrix, cam_params, pt_meshes):
    """Rasterize ``pt_meshes`` and return a per-pixel normalized depth map."""
    focal_x, _, cam_size = cam_params['focal_x'], cam_params['focal_y'], cam_params['size']
    R, T = cam_matrix[:, :3, :3], cam_matrix[:, :3, 3]

    raster_settings = RasterizationSettings(
        image_size=cam_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )

    cameras_perspective = FoVPerspectiveCameras(
        R=R, T=T,
        fov=2 * math.atan(1.0 / focal_x) * 180 / math.pi,  # field of view in degrees
        znear=0.01, zfar=100.0,
        device=pt_meshes.device,
    )

    rasterizer = MeshRasterizer(
        cameras=cameras_perspective,
        raster_settings=raster_settings,
    )
    fragments = rasterizer(pt_meshes)
    z_buffer = fragments.zbuf  # raw depth buffer, shape (bs, H, W, 1)

    # Replace background pixels (depth < 0) with the per-batch maximum depth.
    max_values = torch.amax(z_buffer.clamp_min(0), dim=(1, 2, 3), keepdim=True)
    mask = z_buffer < 0
    depth = torch.where(mask, max_values, z_buffer)

    # Normalize each batch into [0, 1].
    min_vals = depth.amin(dim=(1, 2), keepdim=True)
    max_vals = depth.amax(dim=(1, 2), keepdim=True)
    normalized_depth = (depth - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized_depth


def get_normlized_z_buffer(cam_matrix, cam_params, pt_meshes, ndc, z_buffer_size=512):
    """
    Render a normalized z-buffer for ``pt_meshes`` and check, for each NDC point,
    whether it is occluded by the mesh.

    Args:
        cam_matrix: [batch_size, 4, 4]
        cam_params: dict with focal_x, focal_y, size
        pt_meshes: pytorch3d ``Meshes`` instance
        ndc: [batch_size, num_points, 3], NDC coordinates of the query points
        z_buffer_size: size of the z-buffer image

    Returns:
        normalized_z_buffer_map: [batch_size, 1, H, W]
        occluded_mask: [batch_size, num_points] boolean mask
    """
    normalized_z_buffer_map = render_depth_map(cam_matrix, cam_params, pt_meshes)

    batch_size = ndc.shape[0]
    ndc[:, :, :2] = (ndc[:, :, :2] * 0.5 + 0.5) * z_buffer_size
    ndc_d_min = torch.amin(ndc[:, :, -1], dim=1, keepdim=True)
    ndc_d_max = torch.amax(ndc[:, :, -1], dim=1, keepdim=True)

    ndc[:, :, -1] -= ndc_d_min
    ndc[:, :, -1] /= (ndc_d_max - ndc_d_min + 1e-8)
    ndc[:, :, -1] *= z_buffer_size

    z_buffer_map = normalized_z_buffer_map * z_buffer_size

    u = ndc[:, :, 0].long()
    v = ndc[:, :, 1].long()

    batch_idx = torch.arange(batch_size, device=ndc.device).unsqueeze(1).expand(-1, ndc.shape[1])
    valid_mask = (u >= 0) & (u < z_buffer_size) & (v >= 0) & (v < z_buffer_size)
    z_values = z_buffer_map[
        batch_idx[valid_mask],
        v[valid_mask],
        u[valid_mask],
        0,
    ]

    occluded_mask = torch.zeros(
        (batch_size, ndc.shape[1]), dtype=torch.bool, device=ndc.device,
    )
    occluded_mask[valid_mask] = ndc[:, :, 2][valid_mask] >= z_values
    occluded_mask[~valid_mask] = True

    normalized_z_buffer_map = normalized_z_buffer_map[..., 0].unsqueeze(1)
    return normalized_z_buffer_map, occluded_mask


def render_gaussian(gs_params, cam_matrix, cam_params=None, sh_degree=0, bg_color=None):
    # Build params
    batch_size = cam_matrix.shape[0]
    focal_x, focal_y, cam_size = cam_params['focal_x'], cam_params['focal_y'], cam_params['size']
    points, colors, opacities, scales, rotations = (
        gs_params['xyz'], gs_params['colors'], gs_params['opacities'],
        gs_params['scales'], gs_params['rotations'],
    )
    view_mat, proj_mat, cam_pos = build_camera_matrices(cam_matrix, focal_x, focal_y)
    bg_color = (
        cam_matrix.new_zeros(batch_size, NUM_CHANNELS, dtype=torch.float32)
        if bg_color is None else bg_color
    )
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D
    # (screen-space) means.
    means2D = torch.zeros_like(points, dtype=points.dtype, requires_grad=True, device="cuda") + 0
    try:
        means2D.retain_grad()
    except Exception:
        pass
    # Run rendering
    all_rendered, all_radii = [], []
    for bid in range(batch_size):
        raster_settings = GaussianRasterizationSettings(
            sh_degree=sh_degree, bg=bg_color,
            image_height=cam_size[0], image_width=cam_size[1],
            tanfovx=1.0 / focal_x, tanfovy=1.0 / focal_y,
            viewmatrix=view_mat[bid], projmatrix=proj_mat[bid], campos=cam_pos[bid],
            scale_modifier=1.0, prefiltered=False, debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered, radii = rasterizer(
            means3D=points[bid], means2D=means2D[bid],
            shs=None, colors_precomp=colors[bid],
            opacities=opacities[bid], scales=scales[bid],
            rotations=rotations[bid], cov3D_precomp=None,
        )
        all_rendered.append(rendered)
        all_radii.append(radii)
    all_rendered = torch.stack(all_rendered, dim=0)
    all_radii = torch.stack(all_radii, dim=0)
    return {
        "images": all_rendered, "radii": all_radii, "viewspace_points": means2D,
    }


def remove_faces_with_vertices(faces: torch.Tensor, exclude_vertices: list) -> torch.Tensor:
    if len(exclude_vertices) == 0:
        return faces.clone()

    exclude_tensor = torch.tensor(exclude_vertices, dtype=faces.dtype, device=faces.device)
    faces_flat = faces.squeeze(0)  # [n, 3]

    # Drop any face that contains a vertex listed in ``exclude_tensor``.
    mask = torch.isin(faces_flat, exclude_tensor).any(dim=1)  # [n]
    new_faces_flat = faces_flat[~mask]
    return new_faces_flat.unsqueeze(0)


def build_camera_matrices(cam_matrix, focal_x, focal_y):
    def get_projection_matrix(fov_x, fov_y, z_near=0.01, z_far=100, device='cpu'):
        K = torch.zeros(4, 4, device=device)
        z_sign = 1.0
        K[0, 0] = 1.0 / math.tan((fov_x / 2))
        K[1, 1] = 1.0 / math.tan((fov_y / 2))
        K[3, 2] = z_sign
        K[2, 2] = z_sign * z_far / (z_far - z_near)
        K[2, 3] = -(z_far * z_near) / (z_far - z_near)
        return K

    def get_world_to_view_matrix(transforms):
        assert transforms.shape[-2:] == (3, 4)
        viewmatrix = transforms.new_zeros(transforms.shape[0], 4, 4)
        for i in range(4):
            viewmatrix[:, i, i] = 1.0
        viewmatrix[:, :3, :3] = transforms[:, :3, :3]
        viewmatrix[:, 3, :3] = transforms[:, :3, 3]
        viewmatrix[:, :, :2] *= -1.0
        return viewmatrix

    def get_full_projection_matrix(viewmatrix, fov_x, fov_y):
        proj_matrix = get_projection_matrix(fov_x, fov_y, device=viewmatrix.device)
        full_proj_matrix = viewmatrix @ proj_matrix.transpose(0, 1)
        return full_proj_matrix

    fov_x = 2 * math.atan(1.0 / focal_x)
    fov_y = 2 * math.atan(1.0 / focal_y)
    view_matrix = get_world_to_view_matrix(cam_matrix)
    full_proj_matrix = get_full_projection_matrix(view_matrix, fov_x, fov_y)
    cam_pos = cam_matrix[:, :3, 3]
    return view_matrix, full_proj_matrix, cam_pos
