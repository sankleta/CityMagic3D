import numpy as np
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, SiglipModel

from instance_masks_from_images.utils import mask_and_crop_image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def load_clip(clip_model):
    clip = {"processor": AutoProcessor.from_pretrained(clip_model),
            "model": SiglipModel.from_pretrained(clip_model).to(DEVICE)}
    return clip


def extract_clip_features(clip, image, mask):
    masked_image = mask_and_crop_image(image, mask)

    with torch.no_grad():
        inputs = clip["processor"](images=masked_image, return_tensors="pt").to(DEVICE)
        masked_image_features = clip["model"].get_image_features(**inputs)
    return masked_image_features.cpu().detach().numpy()


def get_query_embedding(clip, text_query):
    text_input_processed = clip["processor"].tokenize(text_query).to(DEVICE)
    with torch.no_grad():
        sentence_embedding = clip["model"].encode_text(text_input_processed)

    sentence_embedding_normalized = (
            sentence_embedding / sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
    return sentence_embedding_normalized.squeeze().numpy()


def compute_similarity_scores(mask_embedding, query_embedding):
    scores = np.zeros(len(mask_embedding))
    for mask_idx, mask_emb in enumerate(mask_embedding):
        mask_norm = np.linalg.norm(mask_emb)
        if mask_norm < 0.001:
            continue
        normalized_emb = (mask_emb / mask_norm)
        scores[mask_idx] = normalized_emb @ query_embedding

    return scores
