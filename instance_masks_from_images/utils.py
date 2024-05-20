import logging

import hydra
import numpy as np
from PIL import Image

from processing import BlocksExchange_xml_parser
from instance_masks_from_images.scene import Camera


logger = logging.getLogger(__name__)


def output_dir():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    return output_dir


def get_extrinsic_matrix(rotation_matrix, center):
    rot = np.eye(4)
    rot[:3, :3] = rotation_matrix
    position_m = np.eye(4)
    position_m[:3, 3:] = -center.reshape((3, 1))
    return rot @ position_m


def load_image_info(cfg):
    intrinsic_matrix, poses_for_images, width, height = BlocksExchange_xml_parser.parse_xml(cfg.scene.cam_info_path)
    camera = Camera(intrinsic_matrix, width, height)
    return camera, poses_for_images


def mask_and_crop_image(orig_image, mask):
    mask_img = Image.fromarray((mask["segmentation"] * 255).astype(np.uint8))
    mask_img = mask_img.resize(orig_image.size, Image.Resampling.NEAREST)
   # mask_img.show(title="Mask")
    # masked_image = Image.composite(orig_image, Image.new("RGB", orig_image.size), mask_img)
    resize_factor = orig_image.size[0] / mask["segmentation"].shape[1]
    logger.debug(f"Resize factor: {resize_factor}")
    x, y, width, height = mask['bbox']
    x = int(x * resize_factor)
    y = int(y * resize_factor)
    width = int(width * resize_factor)
    height = int(height * resize_factor)
    # masked_image = masked_image.crop((x, y, x + width, y + height))
    masked_image = orig_image.crop((x, y, x + width, y + height))
   # image.show(title="Original Image")
   # masked_image.show(title="Masked Image")
    return masked_image
