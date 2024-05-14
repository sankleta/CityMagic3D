import hydra
import numpy as np

from processing import BlocksExchange_xml_parser
from .scene import Camera


def output_dir():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    return output_dir

def get_extrinsic_matrix(rotation_matrix, center):
    rot = np.eye(4)
    rot[:3, :3] = rotation_matrix
    position_m = np.eye(4)
    position_m[:3,3:] = -center.reshape((3,1))
    return rot @ position_m


def load_image_info(cfg):
    intrinsic_matrix, poses_for_images, width, height = BlocksExchange_xml_parser.parse_xml(cfg.scene.cam_info_path)
    camera = Camera(intrinsic_matrix, width, height)
    return camera, poses_for_images
