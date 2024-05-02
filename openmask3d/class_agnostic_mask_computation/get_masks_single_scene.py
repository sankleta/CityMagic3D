from dataclasses import dataclass, astuple
import os
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from trainer.trainer import InstanceSegmentation
from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from utils.point_cloud_utils import splitPointCloud
import open3d as o3d
import numpy as np
import torch
import time
import psutil


def load_model(cfg: DictConfig):
    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")
    model.to(device)
    return model


def load_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    pcd.estimate_normals()
    coords = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    return coords, colors, normals


@dataclass
class SceneData:
    coordinates: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    scene_name: str
    raw_colors: np.ndarray
    raw_normals: np.ndarray
    raw_coordinates: np.ndarray


def process_file(filepath) -> SceneData:
    coords, colors, normals = load_ply(filepath)
    raw_coordinates = coords.copy()
    raw_colors = (colors*255).astype(np.uint8)
    raw_normals = normals

    features = colors
    if len(features.shape) == 1:
        features = np.hstack((features[None, ...], coords))
    else:
        features = np.hstack((features, coords))

    filename = filepath.split("/")[-1][:-4]
    return SceneData(coords, features, [], filename, raw_colors, raw_normals, raw_coordinates)


@hydra.main(config_path="conf", config_name="config_base_class_agn_masks_single_scene.yaml")
def get_class_agnostic_masks(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    load_dotenv(".env")  # is this needed?

    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    model = load_model(cfg)
    model.eval()

    c_fn = hydra.utils.instantiate(cfg.data.test_collation) #(model.config.data.test_collation)

    start = time.time()
    scene_data = process_file(cfg.general.scene_path)
    if cfg.data.do_slicing:
        print("Slicing the point cloud...")
        size = cfg.data.slicing_size
        stride = cfg.data.slicing_stride
        for slice, cond, x, y in splitPointCloud(scene_data.coordinates, size, stride):
            print(psutil.virtual_memory())
            if cond.sum() == 0:
                continue
            labels = np.zeros((0,)) if len(scene_data.labels) == 0 else scene_data.labels[cond]
            slice_data = SceneData(slice,
                                   scene_data.features[cond],
                                   labels, 
                                   scene_data.scene_name + f"_slice_{x}_{y}", 
                                   scene_data.raw_colors[cond], 
                                   scene_data.raw_normals[cond], 
                                   scene_data.raw_coordinates[cond])
            print(f"Processing slice:{slice_data.scene_name} with shape: {slice_data.coordinates.shape}")
            input_batch = [astuple(slice_data) + (0, )]
            batch = c_fn(input_batch)
            with torch.no_grad():
                res_dict = model.get_masks_single_scene(batch)
                del(res_dict)

    else:
        input_batch = [astuple(scene_data) + (0, )]
        batch = c_fn(input_batch)
        with torch.no_grad():
            res_dict = model.get_masks_single_scene(batch)

    end = time.time()

    print("Time elapsed: ", end - start)


@hydra.main(config_path="conf", config_name="config_base_class_agn_masks_single_scene.yaml")
def main(cfg: DictConfig):
    get_class_agnostic_masks(cfg)


if __name__ == "__main__":
    main()
