import numpy as np
import pyvista as pv

import demo.quering
from demo.utils import show_top_20_instances
from instance_masks_from_images import image_text

#point_cloud = pv.read(r"C:\Users\sankl\Downloads\WMSC\WMSC_points.ply")
# loaded_masks = np.load(r'C:\Users\sankl\PycharmProjects\CityMagic3D\instance_masks_from_images\outputs\instance_masks_from_images\2024-05-19_21-46-05\DJI_0887.JPG__mask_indices.npz')
# loaded_text = np.load(r'C:\Users\sankl\PycharmProjects\CityMagic3D\instance_masks_from_images\outputs\instance_masks_from_images\2024-05-19_21-46-05\DJI_0887.JPG__mask_text_embeddings.npz')

mesh = pv.read(r"C:\Users\sankl\Downloads\RA\RA_1M.ply")
loaded_masks = np.load(r'C:\Users\sankl\Downloads\merged_masks.npz')
loaded_text = np.load(r'C:\Users\sankl\Downloads\merged_embeddings.npz')

mask_embeddings = {key: loaded_masks[key] for key in loaded_masks}

mask_text_embeddings = {key: loaded_text[key].reshape(768, ) for key in loaded_text}
query = "green"
model = image_text.load_image_text_model("google/siglip-base-patch16-224")
query_embeddings = demo.quering.get_query_embedding(model, query)

scores = demo.quering.compute_cosine_similarity_scores(mask_text_embeddings, query_embeddings)

# Create a plotter object
plotter = pv.Plotter()
plotter.add_camera_orientation_widget()

# Add the point cloud or mesh to the plotter
plotter.add_mesh(mesh, scalars='RGBA', rgb=True)
plotter.set_color_cycler(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue'])
show_top_20_instances(plotter, mesh, scores, mask_embeddings)

# Display the plot
plotter.show()