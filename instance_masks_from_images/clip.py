from PIL import Image
import requests
import torch
from transformers import AutoProcessor, SiglipModel

from instance_masks_from_images.utils import crop_image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def load_clip(clip_model):
    clip = {"processor": AutoProcessor.from_pretrained(clip_model),
            "model": SiglipModel.from_pretrained(clip_model).to(DEVICE)}
    return clip


def extract_clip_features(clip, image, mask):
    masked_image = crop_image(image, mask)
    with torch.no_grad():
        inputs = clip["processor"](images=masked_image, return_tensors="pt").to(DEVICE)
        masked_image_features = clip["model"].get_image_features(**inputs)
    return masked_image_features

# load image
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# clip_features = extract_clip_features(image)
# print(clip_features)
