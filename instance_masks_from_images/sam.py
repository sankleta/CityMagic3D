from matplotlib import pyplot as plt
import numpy as np
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
        plt.close()
    else:
        plt.show()


