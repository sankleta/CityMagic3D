from PIL import Image
import requests
import torch
from transformers import AutoProcessor, SiglipModel


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device)


def extract_features_siglip(image):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
    return image_features


# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
clip_features = extract_features_siglip(image)
print(clip_features)
