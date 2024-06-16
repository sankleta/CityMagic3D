import os
import torch
from zoedepth_utils import save_raw_16bit, colorize
from PIL import Image
import ssl

'''This is a script with ZoeDepth metric depth estimator. 
Some people swore by it, but it looks like it doesn't generalize well for outdoor scene'''

# Set the directory for input images and output depth maps
INPUT_PATH = r"C:\Users\sankl\Downloads\RA\RA"
OUTPUT_PATH = r"C:\Users\sankl\Downloads\RA\ZoeDepth"

# Setup SSL context
context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
context.load_default_certs()

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

# Load the model from torch hub
repo = "isl-org/ZoeDepth"
# Zoe_N
# model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

# Zoe_K
model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True)
#
# Zoe_NK
# model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

# Check for GPU support
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_zoe_k = model_zoe_k.to(DEVICE)

# Iterate over each image in the directory
for filename in os.listdir(INPUT_PATH):
    if filename.lower().endswith(('.jpg', '.jpeg')):
        path_to_image = os.path.join(INPUT_PATH, filename)
        output_path = os.path.join(OUTPUT_PATH, os.path.splitext(filename)[0] + ".png")
        output_path_colored = os.path.join(OUTPUT_PATH, os.path.splitext(filename)[0] + "_colored.png")

        # Open and process the image
        image = Image.open(path_to_image).convert("RGB")
        depth_tensor = model_zoe_k.infer_pil(image, output_type="tensor")

        # Save the raw depth map
        save_raw_16bit(depth_tensor, output_path)

        # # Colorize and save the colored depth map
        # colored = colorize(depth_tensor)
        # Image.fromarray(colored).save(output_path_colored)

print("Depth map generation complete.")
