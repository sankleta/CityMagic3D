import gc
import logging
import os
import random
import time

import hydra
import json
import numpy as np
from omegaconf import DictConfig
from PIL import Image
import torch

from instance_masks_from_images.image_text import load_image_text_model, extract_text_features
from instance_masks_from_images.utils import get_extrinsic_matrix, load_image_info, output_dir
from instance_masks_from_images.sam import load_sam_mask_generator, show_masks
from instance_masks_from_images.scene import Scene

logger = logging.getLogger(__name__)


def generate_sam_masks(cfg, img_name, img, sam_mask_generator):
    img_arr = np.array(img)
    with torch.no_grad():
        image_masks = sam_mask_generator.generate(img_arr)
    image_masks = [mask for mask in image_masks if mask["area"] > cfg.sam_mask_gen_params.min_mask_region_area]
    logger.info(f"Generated {len(image_masks)} masks for image {img_name}")
    if cfg.debug:
        show_masks(image_masks, img_arr, os.path.join(output_dir(), f"{img_name}__mask_visu.png"))
    save_masks_as_npz(image_masks, os.path.join(output_dir(), f"{img_name}__masks.npz"))
    return image_masks


def save_masks_as_npz(masks, output_path):
    masks_np = np.array([mask['segmentation'] for mask in masks])
    np.savez(output_path, masks=masks_np)


def get_indices_on_point_cloud(resized_resolution, projected_points, visibility_mask, img_mask, camera):
    # get indices of points in the point cloud that project to the mask
    segmentation = img_mask["segmentation"]  # 2D mask, array of shape HW

    projected_visible_points_coords = projected_points[visibility_mask]
    resized_proj_vis_pts_coords = projected_visible_points_coords * resized_resolution / np.array(camera.resolution,
                                                                                                  dtype=np.int16)
    resized_proj_vis_pts_coords = resized_proj_vis_pts_coords.round().astype(int)
    resized_proj_vis_pts_coords = resized_proj_vis_pts_coords.clip(0, resized_resolution - 1)

    visible_points_mask = segmentation[resized_proj_vis_pts_coords[:, 1], resized_proj_vis_pts_coords[:, 0]]

    all_points_mask = np.zeros(len(projected_points), dtype=bool)
    all_points_mask[visibility_mask] = visible_points_mask
    return np.where(all_points_mask)[0].astype(np.uint32)


@hydra.main(version_base="1.3", config_path=".", config_name="config2.yaml")
def main(cfg: DictConfig):
    if cfg.debug:
        logging.basicConfig(level=logging.DEBUG)
    # Load the SAM model
    sam_mask_generator = load_sam_mask_generator(cfg)
    image_text_model = load_image_text_model(cfg.image_text_model)
    images_dir = cfg.scene.images_dir
    files = os.listdir(images_dir)
    if cfg.start_from_img:
        files = files[files.index(cfg.start_from_img):]
    if cfg.end_at_img:
        files = files[:files.index(cfg.end_at_img)]
    if cfg.samples > 0:
        files = random.sample(files, cfg.samples)
    
    # load scene with point cloud, camera info
    scene = Scene(cfg.scene)
    camera, poses_for_images = load_image_info(cfg)

    resized_resolution = np.array([cfg.resized_img_width, cfg.resized_img_height], dtype=np.uint16)

    # generate masks from images and save masks as npz
    for i, img_name in enumerate(files):
        logger.info(f"Processing image {i+1}/{len(files)}: {img_name}")
        img_path = os.path.join(cfg.scene.images_dir, img_name)
        orig_img = Image.open(img_path).convert("RGB")
        logger.info(f"image size: {orig_img.size}")
        img = orig_img.resize((cfg.resized_img_width, cfg.resized_img_height), Image.Resampling.LANCZOS)
        logger.info(f"resized image size: {img.size}")
        image_masks = generate_sam_masks(cfg, img_name, img, sam_mask_generator)

        mask_indices = {}
        mask_text_embeddings = {}

        img_extrinsic = get_extrinsic_matrix(*poses_for_images[img_name])
        inside_mask, visibility_mask, projected_points = scene.get_visible_points(camera, img_extrinsic)

        logger.info("Adding image-text embedding and projecting masks to point cloud...")
        for i, img_mask in enumerate(image_masks):
            logger.debug(f"Mask {i}")
            # Have to remove very small masks (dots and lines), image_text_model doesn't respond well
            if img_mask['bbox'][2] < 2 or img_mask['bbox'][3] < 2:
                logger.info(f"Skipped too small mask {i}")
                continue
            save_masked_image = os.path.join(output_dir(), f"{img_name}__mask_{i}.png") if cfg.debug else None
            text_embedding = extract_text_features(image_text_model, orig_img, img_mask, save_masked_image, cfg.crop_margin)
            mask_indices[str(i)] = get_indices_on_point_cloud(resized_resolution, projected_points, visibility_mask, img_mask, camera)
            mask_text_embeddings[str(i)] = text_embedding

        # save all mask indices
        np.savez(os.path.join(output_dir(), f"{img_name}__mask_indices.npz"), **mask_indices)
        np.savez(os.path.join(output_dir(), f"{img_name}__mask_text_embeddings.npz"), **mask_text_embeddings)

        for img_mask in image_masks:
            del (img_mask["segmentation"])
        gc.collect()
        torch.cuda.empty_cache()

        # save mask metadata
        with open(os.path.join(output_dir(), f"{img_name}__mask_metadata.json"), "a") as f:
            for img_mask in image_masks:
                json.dump(img_mask, f)
                f.write("\n")

    # TODO visualize

    # TODO: merging masks corresponding to the same object (in a second phase)


if __name__ == "__main__":
    logger.info("Starting instance masks generation...")
    start = time.time()
    main()
    logger.info(f"Instance masks generation finished in {(time.time() - start) / 60} minutes.")
