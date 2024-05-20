import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")


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
