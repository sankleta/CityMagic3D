import torch
from transformers import AutoProcessor, SiglipModel

from instance_masks_from_images.utils import mask_and_crop_image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def load_image_text_model(model_id):
    image_text_model = {"processor": AutoProcessor.from_pretrained(model_id),
                        "model": SiglipModel.from_pretrained(model_id).to(DEVICE)}
    return image_text_model


def extract_text_features(image_text_model, image, mask):
    masked_image = mask_and_crop_image(image, mask)

    with torch.no_grad():
        inputs = image_text_model["processor"](images=masked_image, return_tensors="pt").to(DEVICE)
        masked_image_features = image_text_model["model"].get_image_features(**inputs)
    return masked_image_features.cpu().detach().numpy()
