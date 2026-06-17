#!/usr/bin/env python
# Author: Jianiang Ren

import os
import argparse

import numpy as np
import torch
import torchvision
import lightning
from tqdm.rich import tqdm

from core.data import DriverData
from core.models.omg_model import OMGAvatar
from core.libs.utils import ConfigDict
from core.libs.GAGAvatar_track.engines import CoreEngine as TrackEngine


IMAGE_EXTENSIONS = ('jpg', 'png', 'jpeg')
TRACKED_DIR = 'render_results/tracked'
TRACKED_PT_PATH = os.path.join(TRACKED_DIR, 'tracked.pt')
WATER_MARK_PATH = 'assets/water_mark/water_mark.png'

def is_image(image_path):
    return image_path.split('.')[-1].lower() in IMAGE_EXTENSIONS


def load_model(resume_path, device):
    print('Loading model...')
    lightning_fabric = lightning.Fabric(accelerator=device, strategy='auto', devices=[0])
    lightning_fabric.launch()
    full_checkpoint = lightning_fabric.load(resume_path)
    print("full checkpoint keys:", full_checkpoint.keys())
    meta_cfg = ConfigDict(init_dict=full_checkpoint['meta_cfg'])
    model = OMGAvatar(model_cfg=meta_cfg.MODEL)
    model.load_state_dict(full_checkpoint['model'], strict=False)
    model = lightning_fabric.setup(model)
    model.eval()
    print(str(meta_cfg))
    return model, meta_cfg, lightning_fabric


def build_driver_dataloader(driver_path, feature_data, meta_cfg, track_engine, force_retrack):
    if os.path.isdir(driver_path):
        driver_name = os.path.basename(driver_path)
        driver_dataset = DriverData(driver_path, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
        num_workers = 8
    else:
        driver_name = os.path.basename(driver_path).split('.')[0]
        driver_data = get_tracked_results(driver_path, track_engine, force_retrack=force_retrack)
        if driver_data is None:
            return None, driver_name
        driver_dataset = DriverData({driver_name: driver_data}, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
        num_workers = 2
    driver_dataloader = torch.utils.data.DataLoader(
        driver_dataset, batch_size=1, num_workers=num_workers, shuffle=False,
    )
    return driver_dataloader, driver_name



def load_water_mark(path, device):
    """Load an RGBA water-mark image as a float tensor in [0, 1] on ``device``."""
    water_mark = torchvision.io.read_image(
        path, mode=torchvision.io.ImageReadMode.RGB_ALPHA,
    ).float() / 255.0
    return water_mark.to(device)


def add_water_mark(image, water_mark):
    """Alpha-blend ``water_mark`` onto the bottom-right corner of ``image``.

    Args:
        image: tensor of shape (B, 3, H, W), values in [0, 1].
        water_mark: tensor of shape (4, h, w), RGBA, values in [0, 1].

    Returns:
        A new tensor with the water mark composited onto the bottom-right corner.
    """
    # Align water_mark's device and dtype with image to avoid mismatches
    # caused by mixed-precision / multi-device setups.
    water_mark = water_mark.to(device=image.device, dtype=image.dtype)
    h, w = water_mark.shape[-2], water_mark.shape[-1]

    _water_mark_rgb = water_mark[None, :3]
    _water_mark_alpha = water_mark[None, 3:4].expand(-1, 3, -1, -1) * 0.8

    # Clone to avoid writing back into a non-contiguous / read-only view.
    image = image.clone()
    _mark_patch = image[..., -h:, -w:]
    _mark_patch = _mark_patch * (1 - _water_mark_alpha) + _water_mark_rgb * _water_mark_alpha
    image[..., -h:, -w:] = _mark_patch
    return image

def save_results(images, feature_data, driver_dataset, dump_dir, feature_name, driver_name, subdivide_times):
    os.makedirs(dump_dir, exist_ok=True)
    if driver_dataset._is_video:
        dump_path = os.path.join(dump_dir, f'{feature_name}_{driver_name}_sub{subdivide_times}.mp4')
        merged_images = torch.stack(images)
        feature_images = torch.stack([feature_data['image']] * merged_images.shape[0])
        merged_images = torch.cat([feature_images, merged_images], dim=-1)
        merged_images = (merged_images * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(dump_path, merged_images, fps=25.0)
    else:
        dump_path = os.path.join(dump_dir, f'{feature_name}_{driver_name}.jpg')
        merged_images = torchvision.utils.make_grid(images, nrow=5, padding=0)
        feature_images = torchvision.utils.make_grid(
            [feature_data['image']] * (merged_images.shape[-2] // 512), nrow=1, padding=0,
        )
        merged_images = torch.cat([feature_images, merged_images], dim=-1)
        torchvision.utils.save_image(merged_images, dump_path)
    return dump_path


def inference(image_path, 
            driver_path, 
            resume_path, 
            force_retrack=False, 
            subdivide_times=2, 
            device='cuda', 
            enhance_shoulder=False):
    lightning.fabric.seed_everything(42)
    driver_path = driver_path[:-1] if driver_path.endswith('/') else driver_path

    # load model
    model, meta_cfg, lightning_fabric = load_model(resume_path, device)
    track_engine = TrackEngine(focal_length=12.0, device=device)
    water_mark = load_water_mark(WATER_MARK_PATH, device)

    # build input data
    feature_name = os.path.basename(image_path).split('.')[0]
    feature_data = get_tracked_results(image_path, track_engine, force_retrack=force_retrack)
    if feature_data is None:
        print(f'Finish inference, no face in input: {image_path}.')
        return

    # build driver data
    driver_dataloader, driver_name = build_driver_dataloader(
        driver_path, feature_data, meta_cfg, track_engine, force_retrack,
    )
    if driver_dataloader is None:
        print(f'Finish inference, no face in driver: {driver_path}.')
        return
    driver_dataset = driver_dataloader.dataset
    driver_dataloader = lightning_fabric.setup_dataloaders(driver_dataloader)

    # configure model
    model._gs_params = None
    model.set_subdivide_times(subdivide_times)
    model.use_neural_renderer = True

    # run inference
    images = []
    total_fps = 0
    total_frame = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(driver_dataloader)):
            render_results = model(batch, is_training=False, enhance_shoulder=enhance_shoulder)

            gt_rgb = render_results['t_image'].clamp(0, 1)
            pred_sr_rgb = render_results['sr_gen_image'].clamp(0, 1)
            pred_sr_rgb = add_water_mark(pred_sr_rgb, water_mark)

            if idx > 1 and total_frame < 100:
                total_fps += render_results.get('fps')
                total_frame += 1

            visulize_rgbs = torchvision.utils.make_grid([gt_rgb[0], pred_sr_rgb[0]], nrow=4, padding=0)
            images.append(visulize_rgbs.cpu())

            if idx > 600:
                break

    print('total frame:', total_frame)
    print(f'total fps: {total_fps}.')
    print(f'Average FPS: {total_fps / (total_frame + 1.0e-5)}.')

    # save results
    dump_dir = os.path.join('render_results', meta_cfg.MODEL.NAME.split('_')[0])
    dump_path = save_results(images, feature_data, driver_dataset, dump_dir, 
                            feature_name, driver_name, subdivide_times)
    print(f'Finish inference: {dump_path}.')


def get_tracked_results(image_path, track_engine, force_retrack=False):
    if not is_image(image_path):
        print(f'Please input a image path, got {image_path}.')
        return None
    if not os.path.exists(TRACKED_PT_PATH):
        os.makedirs(TRACKED_DIR, exist_ok=True)
        torch.save({}, TRACKED_PT_PATH)
    tracked_data = torch.load(TRACKED_PT_PATH, weights_only=False)
    image_base = os.path.basename(image_path)

    if image_base in tracked_data and not force_retrack:
        print(f'Load tracking result from cache: {TRACKED_PT_PATH}.')
    else:
        print(f'Tracking {image_path}...')
        image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).float()
        feature_data = track_engine.track_image([image], [image_path])
        if feature_data is None:
            print(f'No face detected in {image_path}.')
            return None
        feature_data = feature_data[image_path]
        torchvision.utils.save_image(
            torch.tensor(feature_data['vis_image']),
            os.path.join(TRACKED_DIR, '{}.jpg'.format(image_base.split('.')[0])),
        )
        tracked_data[image_base] = feature_data

        # track all images in this folder
        other_names = [i for i in os.listdir(os.path.dirname(image_path)) if is_image(i)]
        other_paths = [os.path.join(os.path.dirname(image_path), i) for i in other_names]
        if len(other_paths) <= 35:
            print('Track on all images in this folder to save time.')
            other_images = [
                torchvision.io.read_image(imp, mode=torchvision.io.ImageReadMode.RGB).float()
                for imp in other_paths
            ]
            try:
                other_feature_data = track_engine.track_image(other_images, other_names)
                for key in other_feature_data:
                    torchvision.utils.save_image(
                        torch.tensor(other_feature_data[key]['vis_image']),
                        os.path.join(TRACKED_DIR, '{}.jpg'.format(key.split('.')[0])),
                    )
                tracked_data.update(other_feature_data)
            except Exception as e:
                print(f'Error: {e}.')
        # save tracking result
        torch.save(tracked_data, TRACKED_PT_PATH)

    feature_data = tracked_data[image_base]
    for key in list(feature_data.keys()):
        if isinstance(feature_data[key], np.ndarray):
            feature_data[key] = torch.tensor(feature_data[key])
    return feature_data


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter('ignore', category=TqdmExperimentalWarning, lineno=0, append=False)

    # build args
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', required=True, type=str)
    parser.add_argument('--driver_path', '-d', required=True, type=str)
    parser.add_argument('--subdivide_times', '-sub', default=2, type=int)
    parser.add_argument('--force_retrack', '-f', action='store_true')
    parser.add_argument('--shoulder_enhance', '-sh', action='store_true')
    parser.add_argument('--resume_path', '-r', default='./ckpts/omg_ckpt.pt', type=str)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    inference(
        args.image_path,
        args.driver_path,
        args.resume_path,
        args.force_retrack,
        subdivide_times=args.subdivide_times,
        enhance_shoulder=args.shoulder_enhance,
    )
