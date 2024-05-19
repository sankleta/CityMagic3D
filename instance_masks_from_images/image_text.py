import numpy as np
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


def get_query_embedding(image_text_model, text_query):
    text_input_processed = image_text_model["processor"].tokenize(text_query).to(DEVICE)
    with torch.no_grad():
        sentence_embedding = image_text_model["model"].encode_text(text_input_processed)

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
