import numpy as np
import pyvista as pv

from demo.utils import show_top_n_instances, show_heatmap, clean
from instance_masks_from_images import image_text



#TEXT_EMBEDDING_SIZE = 768
TEXT_EMBEDDING_SIZE = 512

MESH = pv.read(r"C:\Users\sankl\Downloads\RA\RA_1M.ply")
LOADED_MASKS = np.load(r'C:\Users\sankl\PycharmProjects\CityMagic3D\merge_masks\outputs\merge_masks\2024-05-25_00-11-29\merged_masks.npz')
LOADED_TEXT_EMBEDDINGS = np.load(r'C:\Users\sankl\PycharmProjects\CityMagic3D\merge_masks\outputs\merge_masks\2024-05-25_00-11-29\merged_embeddings__avg.npz')

#QUERY = "palm tree"
#QUERY = "red car on a parking lot"
QUERY = "roof of a residential house"

mask_embeddings = {key: LOADED_MASKS[key] for key in LOADED_MASKS}

mask_text_embeddings = {key: LOADED_TEXT_EMBEDDINGS[key].reshape(TEXT_EMBEDDING_SIZE, ) for key in
                        LOADED_TEXT_EMBEDDINGS}

clean(mask_embeddings, mask_text_embeddings)

model = image_text.load_image_text_model("openai/clip-vit-base-patch32")
query_embeddings = image_text.get_query_embedding(model, QUERY)

scores = image_text.compute_cosine_similarity_scores(mask_text_embeddings, query_embeddings)

# Create a plotter object
plotter = pv.Plotter()
# plotter.subplot(0, 0)
plotter.add_text(f"Query: '{QUERY}'", font_size=14)

# # Add the mesh to the plotter
# plotter.add_mesh(MESH, scalars='RGBA', rgb=True)
# show_heatmap(plotter, MESH, scores, mask_embeddings)
#
# plotter.subplot(0, 1)
plotter.add_mesh(MESH, scalars='RGBA', rgb=True)
# Add masks as points on top
show_top_n_instances(plotter, MESH, scores, mask_embeddings, 20)
plotter.show()

