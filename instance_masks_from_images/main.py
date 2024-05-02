import gc
import os
import random

import hydra
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch


def load_sam_mask_generator(cfg):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")
    sam = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_checkpoint)
    sam.to(device)
    sam_mask_generator = SamAutomaticMaskGenerator(model=sam, **cfg.sam_mask_gen_params)
    return sam_mask_generator


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_masks(masks, image, output_path=None):
    plt.figure(figsize=(40, 30))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    if output_path:
        plt.savefig(output_path)
    # plt.show()


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    sam_mask_generator = load_sam_mask_generator(cfg)
    input_dir = cfg.input_dir
    files = random.sample(os.listdir(input_dir), cfg.samples)
    os.makedirs(cfg.output_dir, exist_ok=True)
    for img_path in files:
        img_path = os.path.join(input_dir, img_path)
        img = Image.open(img_path).convert("RGB")
        print(f"image size: {img.size}")
        img = img.resize((cfg.resized_img_width, cfg.resized_img_height), Image.Resampling.LANCZOS)
        print(f"resized image size: {img.size}")
        img_arr = np.array(img)
        with torch.no_grad():
            masks = sam_mask_generator.generate(img_arr)
        print(f"Generated {len(masks)} masks for image {img_path}")
        out_img_file_name = os.path.basename(img_path).split(".")[0] + "_mask.png"
        show_masks(masks, img_arr, os.path.join(cfg.output_dir, out_img_file_name))
        del(masks)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        torch.cuda.empty_cache()

