import torch
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, SiglipModel

from instance_masks_from_images.utils import mask_and_crop_image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def load_image_text_model(model_id):
    if "siglip" in model_id:
        image_text_model = {"processor": AutoProcessor.from_pretrained(model_id),
                    "model": SiglipModel.from_pretrained(model_id).to(DEVICE)}
    else:
        image_text_model = {"processor": CLIPProcessor.from_pretrained(model_id),
                            "model": CLIPModel.from_pretrained(model_id).to(DEVICE)}
    return image_text_model


def extract_text_features(image_text_model, image, mask, save_masked_image=None):
    masked_image = mask_and_crop_image(image, mask)
    if save_masked_image:
        masked_image.save(save_masked_image)

    with torch.no_grad():
        inputs = image_text_model["processor"](images=masked_image, return_tensors="pt").to(DEVICE)
        masked_image_features = image_text_model["model"].get_image_features(**inputs)
    return masked_image_features.cpu().detach().numpy()


def get_query_embedding(image_text_model, text_query):
    text_input_processed = image_text_model["processor"](text=text_query, padding="max_length", return_tensors="pt").to(
        DEVICE)
    with torch.no_grad():
        sentence_embedding = image_text_model["model"].get_text_features(**text_input_processed)

    return sentence_embedding.squeeze().cpu()


def compute_cosine_similarity_scores(mask_embeddings, query_embedding):
    cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-3)
    scores = {}
    for mask_idx in mask_embeddings:
        scores[mask_idx] = cossim(torch.from_numpy(mask_embeddings[mask_idx]), query_embedding)
    return scores
