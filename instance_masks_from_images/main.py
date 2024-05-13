import gc
import logging
import os
import random

import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image
import torch

from processing import BlocksExchange_xml_parser
from .sam import load_sam_mask_generator, show_masks
from .scene import Scene, Camera


logger = logging.getLogger(__name__)


def load_image_info(cfg):
    intrinsic_matrix, poses_for_images, width, height = BlocksExchange_xml_parser.parse_xml(cfg.scene.cam_info_path)
    camera = Camera(intrinsic_matrix, width, height)
    return camera, poses_for_images


def generate_sam_masks(cfg, img_name, sam_mask_generator):
    img_path = os.path.join(cfg.scene.images_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    logger.info(f"image size: {img.size}")
    img = img.resize((cfg.resized_img_width, cfg.resized_img_height), Image.Resampling.LANCZOS)
    logger.info(f"resized image size: {img.size}")
    img_arr = np.array(img)
    with torch.no_grad():
        image_masks = sam_mask_generator.generate(img_arr)
    logger.info(f"Generated {len(image_masks)} masks for image {img_path}")
    show_masks(image_masks, img_arr, os.path.join(cfg.output_dir, f"{img_name}__mask_visu.png"))
    save_masks_as_npz(image_masks, os.path.join(cfg.output_dir, f"{img_name}__masks.npz"))
    return image_masks


def save_masks_as_npz(masks, output_path):
    masks_np = np.array([mask['segmentation'] for mask in masks])
    np.savez(output_path, masks=masks_np)


def get_indices_on_point_cloud(resized_resolution, projected_points, visibility_mask, img_mask, camera):
    # get indices of points in the point cloud that project to the mask
    segmentation = img_mask["segmentation"]  # 2D mask, array of shape HW

    projected_visible_points_coords = projected_points[visibility_mask]
    resized_proj_vis_pts_coords = projected_visible_points_coords * resized_resolution / np.array(camera.resolution, dtype=np.int16)
    resized_proj_vis_pts_coords = resized_proj_vis_pts_coords.round().astype(int)
    resized_proj_vis_pts_coords = resized_proj_vis_pts_coords.clip(0, resized_resolution - 1)
    
    visible_points_mask = segmentation[resized_proj_vis_pts_coords[:, 1], resized_proj_vis_pts_coords[:, 0]]

    all_points_mask = np.zeros(len(projected_points), dtype=bool)
    all_points_mask[visibility_mask] = visible_points_mask
    return np.where(all_points_mask)[0]


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    # Load the SAM model
    sam_mask_generator = load_sam_mask_generator(cfg)
    images_dir = cfg.scene.images_dir
    files = random.sample(os.listdir(images_dir), cfg.samples)
    os.makedirs(cfg.output_dir)
    mask_metadata = {}

    # load scene with point cloud, camera info
    scene = Scene(cfg.scene)
    camera, poses_for_images = load_image_info(cfg)

    resized_resolution = np.array([cfg.resized_img_width, cfg.resized_img_height], dtype=np.uint16)
    
    # generate masks from images and save masks as npz
    for img_name in files:
        image_masks = generate_sam_masks(cfg, img_name, sam_mask_generator)

        mask_indices = {}
        
        img_pose = poses_for_images[img_name]
        inside_mask, visibility_mask, projected_points = scene.get_visible_points(camera, img_pose)

        for i, img_mask in enumerate(image_masks):
            # project masks to point cloud
            mask_indices[str(i)] = get_indices_on_point_cloud(resized_resolution, projected_points, visibility_mask, img_mask, camera)

        # save all mask indices
        np.savez(os.path.join(cfg.output_dir, f"{img_name}__mask_indices.npz"), **mask_indices)            

        for img_mask in image_masks:
            del(img_mask["segmentation"])
        mask_metadata[img_name] = image_masks
        gc.collect()
        torch.cuda.empty_cache()

    # TODO save clip embedding of masks

    # TODO: merging masks corresponding to the same object


if __name__ == "__main__":
    main()

