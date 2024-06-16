from collections import defaultdict
from dataclasses import dataclass
import glob
import logging
import os
import random
import time
from typing import Optional

from omegaconf import DictConfig
import open3d as o3d
import hydra
import networkx as nx
import numpy as np
import pyqtree

from instance_masks_from_images.utils import output_dir, get_extrinsic_matrix
from instance_masks_from_images.scene import Camera
from merge_masks.utils import get_2d_bounding_box_of_point_set
from processing import BlocksExchange_xml_parser

logger = logging.getLogger(__name__)


@dataclass
class MaskInfo:
    img_name: str
    key: str
    point_set: np.ndarray
    embedding: np.ndarray
    embedding_2: Optional[np.ndarray] = None

    @property
    def size(self):
        return len(self.point_set)
    
    @staticmethod
    def is_close(mask_info, other_mask_info, min_intersection_ratio):
        intersection = len(mask_info.point_set.intersection(other_mask_info.point_set))
        return intersection > mask_info.size * min_intersection_ratio \
            or intersection > other_mask_info.size * min_intersection_ratio
    
    @staticmethod
    def is_close_both_ways(mask_info, other_mask_info, min_intersection_ratio):
        intersection = len(mask_info.point_set.intersection(other_mask_info.point_set))
        return intersection > mask_info.size * min_intersection_ratio \
            and intersection > other_mask_info.size * min_intersection_ratio
    
    @staticmethod
    def is_close_iou(mask_info, other_mask_info, min_iou):
        intersection = len(mask_info.point_set.intersection(other_mask_info.point_set))
        union = len(mask_info.point_set.union(other_mask_info.point_set))
        return intersection / union > min_iou
    
    @staticmethod
    def is_close_w_embedding(mask_info, other_mask_info, min_intersection_ratio, min_embedding_similarity):
        cosine_similarity = np.dot(mask_info.embedding, other_mask_info.embedding.T) / \
            (np.linalg.norm(mask_info.embedding) * np.linalg.norm(other_mask_info.embedding))
        return cosine_similarity > min_embedding_similarity and MaskInfo.is_close(mask_info, other_mask_info, min_intersection_ratio)
    
    @staticmethod
    def merge_masks(masks, key):
        merged_mask = set()
        for mask in masks:
            merged_mask.update(mask.point_set)
        merged_embedding_avg = np.average([mask.embedding for mask in masks], axis=0)
        merged_embedding_max = np.max([mask.embedding for mask in masks], axis=0)
        return MaskInfo("merged", 
                        key, 
                        np.array(list(merged_mask), dtype=np.uint32), 
                        merged_embedding_avg,
                        merged_embedding_max)


class ViewIntersectionChecker:
    def __init__(self, cfg: DictConfig):
        intrinsic_matrix, poses_for_images, width, height = BlocksExchange_xml_parser.parse_xml(cfg.cam_info_path)
        self.poses_for_images = poses_for_images
        self.camera = Camera(intrinsic_matrix, width, height)
        self.view_depth = cfg.view_depth
        self.view_pyramids = {}
        self.triangles = o3d.utility.Vector3iVector([[0,1,2], [0,2,3], [0,3,4], [0, 4,1], [1,2,3], [1,3,4]])

    def view_intersect(self, img_name1, img_name2):
        pyramid_1 = self.get_view_pyramid(img_name1)
        pyramid_2 = self.get_view_pyramid(img_name2)
        return pyramid_1.is_intersecting(pyramid_2)
    
    def get_view_pyramid(self, img_name):
        if img_name in self.view_pyramids:
            return self.view_pyramids[img_name]
        cam_lines = self.get_camera_lines(img_name)
        pyramid = o3d.geometry.TriangleMesh(vertices=cam_lines.points, triangles=self.triangles) 
        self.view_pyramids[img_name] = pyramid
        return pyramid
    
    def get_camera_lines(self, img_name):
        rot_m, center = self.poses_for_images[img_name]
        extrinsic = get_extrinsic_matrix(rot_m, center)
        camera_lines = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=self.camera.resolution[0], view_height_px=self.camera.resolution[1], 
            intrinsic=self.camera.intrinsic_matrix.copy(), 
            extrinsic=extrinsic,
            scale=self.view_depth)
        return camera_lines


def build_graph_and_mask_infos__filter_view_intersection(cfg: DictConfig, mask_indices_files: list[str]):
    graph = nx.Graph()
    mask_infos_by_image = defaultdict(dict)
    view_intersection_checker = ViewIntersectionChecker(cfg)

    for i, file in enumerate(mask_indices_files):
        img_name = file.split("__")[0]
        logger.info(f"Loading {file}... {i}/{len(mask_indices_files)}")
        img_masks = np.load(os.path.join(cfg.input_dir, file))
        embedding_file = os.path.join(cfg.input_dir, f"{img_name}__mask_text_embeddings.npz")
        embeddings = np.load(embedding_file)
        for key in img_masks.keys():
            mask = set(img_masks[key])
            if len(mask) < cfg.min_mask_size:
                logger.debug(f"Mask {key} is too small, skipping.")
                continue
            embedding = embeddings[key]
            mask_info = MaskInfo(img_name, key, mask, embedding)
            graph.add_node((img_name, key))
            mask_infos_by_image[img_name][key] = mask_info
            for other_img_name in mask_infos_by_image.keys():
                if other_img_name == img_name:
                    continue
                if not view_intersection_checker.view_intersect(img_name, other_img_name):
                    logger.debug(f"Views {img_name} and {other_img_name} don't intersect, skipping.")
                    continue
                for other_key, other_mask_info in mask_infos_by_image[other_img_name].items():
                    if mask_info.is_close(other_mask_info, cfg.min_intersection_ratio):
                        graph.add_edge((img_name, key), (other_img_name, other_key))
    return graph, mask_infos_by_image


def build_graph_and_mask_infos__quadtree(cfg: DictConfig, mask_indices_files: list[str]):
    graph = nx.Graph()
    mask_infos_by_image = defaultdict(dict)
    quadtree_of_all_masks = pyqtree.Index(bbox=cfg.bbox)
    point_cloud = o3d.io.read_point_cloud(cfg.point_cloud_path)
    is_close = getattr(MaskInfo, cfg.closeness_method)

    for i, file in enumerate(mask_indices_files):
        img_name = file.split("__")[0]
        logger.info(f"Loading {file}... {i}/{len(mask_indices_files)}")
        img_masks = np.load(os.path.join(cfg.input_dir, file))
        embedding_file = os.path.join(cfg.input_dir, f"{img_name}__mask_text_embeddings.npz")
        embeddings = np.load(embedding_file)
        
        for key in img_masks.keys():
            mask = set(img_masks[key])
            if len(mask) < cfg.min_mask_size:
                logger.debug(f"Mask {key} is too small, skipping.")
                continue
            embedding = embeddings[key]
            mask_info = MaskInfo(img_name, key, mask, embedding)
            graph.add_node((img_name, key))
            mask_infos_by_image[img_name][key] = mask_info

            bbox = get_2d_bounding_box_of_point_set(point_cloud, img_masks[key])

            for item in quadtree_of_all_masks.intersect(bbox):
                other_img_name, other_key = item
                other_mask_info = mask_infos_by_image[other_img_name][other_key]
                if is_close(mask_info, other_mask_info, **cfg.closeness_args):
                    graph.add_edge((img_name, key), (other_img_name, other_key))

            quadtree_of_all_masks.insert((img_name, key), bbox)

    logger.info(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)}.")
    return graph, mask_infos_by_image


def merge_by_intersection_ratio(cfg: DictConfig):
    mask_indices_files = glob.glob("*__mask_indices.npz", root_dir=cfg.input_dir)
    if cfg.max_files:
        mask_indices_files = random.sample(mask_indices_files, cfg.max_files)
    logger.info(f"{'Selected' if cfg.max_files else 'Found'} {len(mask_indices_files)} mask indices files.")
    total_size = sum([os.stat(os.path.join(cfg.input_dir, file)).st_size for file in mask_indices_files])
    logger.info(f"Total size of mask indices files: {total_size / 1024**2} MB.")

    graph, mask_infos_by_image = build_graph_and_mask_infos__quadtree(cfg, mask_indices_files)
    
    logger.info(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    merged_mask_infos = {}
    for i, connected_component in enumerate(nx.connected_components(graph)):
        mask_infos_in_component = [mask_infos_by_image[node[0]][node[1]] for node in connected_component]
        merged_mask_info = MaskInfo.merge_masks(mask_infos_in_component, str(i))
        merged_mask_infos[str(i)] = merged_mask_info

    logger.info(f"Merged {len(merged_mask_infos)} masks.")
    merged_masks = {mask_info.key: mask_info.point_set for mask_info in merged_mask_infos.values()}
    np.savez(os.path.join(output_dir(), "merged_masks.npz"), **merged_masks)
    logger.info(f"Saved merged masks to {output_dir()}/merged_masks.npz.")

    merged_embeddings = {mask_info.key: mask_info.embedding for mask_info in merged_mask_infos.values()}
    np.savez(os.path.join(output_dir(), "merged_embeddings__avg.npz"), **merged_embeddings)
    logger.info(f"Saved merged embeddings to {output_dir()}/merged_embeddings__avg.npz.")

    merged_embeddings_max = {mask_info.key: mask_info.embedding_2 for mask_info in merged_mask_infos.values()}
    np.savez(os.path.join(output_dir(), "merged_embeddings__max.npz"), **merged_embeddings_max)
    logger.info(f"Saved merged embeddings to {output_dir()}/merged_embeddings__max.npz.")


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    merge_by_intersection_ratio(cfg)


if __name__ == "__main__":
    logger.info("Starting instance masks merging...")
    start = time.time()
    main()
    logger.info(f"Instance masks merging finished in {(time.time() - start) / 60} minutes.")

