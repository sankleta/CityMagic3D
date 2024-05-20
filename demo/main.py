import numpy as np
import pyvista as pv

from demo.utils import show_top_20_instances, show_heatmap
from instance_masks_from_images import image_text

TEXT_EMBEDDING_SIZE = 768

MESH = pv.read(r"C:\Users\sankl\Downloads\RA\RA_1M.ply")
LOADED_MASKS = np.load(r'C:\Users\sankl\Downloads\merged_masks.npz')
LOADED_TEXT_EMBEDDINGS = np.load(r'C:\Users\sankl\Downloads\merged_embeddings.npz')

QUERY = "green"

mask_embeddings = {key: LOADED_MASKS[key] for key in LOADED_MASKS}

mask_text_embeddings = {key: LOADED_TEXT_EMBEDDINGS[key].reshape(TEXT_EMBEDDING_SIZE, ) for key in LOADED_TEXT_EMBEDDINGS}

model = image_text.load_image_text_model("google/siglip-base-patch16-224")
query_embeddings = image_text.get_query_embedding(model, QUERY)

scores = image_text.compute_cosine_similarity_scores(mask_text_embeddings, query_embeddings)

# Create a plotter object
plotter = pv.Plotter()
plotter.add_text(f"Query: '{QUERY}'", font_size=14)

# Add the mesh to the plotter
plotter.add_mesh(MESH, scalars='RGBA', rgb=True)

# Add masks as points on top
# show_top_20_instances(plotter, mesh, scores, mask_embeddings)
show_heatmap(plotter, MESH, scores, mask_embeddings)

# Display the plot
plotter.show()
