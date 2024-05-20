import numpy as np
import pyvista as pv

from instance_masks_from_images import image_text

point_cloud = pv.read(r"C:\Users\sankl\Downloads\WMSC\WMSC_points.ply")
loaded_masks = np.load(r'C:\Users\sankl\PycharmProjects\CityMagic3D\instance_masks_from_images\outputs\instance_masks_from_images\2024-05-19_21-46-05\DJI_0887.JPG__mask_indices.npz')
loaded_text = np.load(r'C:\Users\sankl\PycharmProjects\CityMagic3D\instance_masks_from_images\outputs\instance_masks_from_images\2024-05-19_21-46-05\DJI_0887.JPG__mask_text_embeddings.npz')


mask_embeddings = {key: loaded_masks[key] for key in loaded_masks}
mask_text_embeddings = {key: loaded_text[key].reshape(768,) for key in loaded_text}
query = "vegetation"
model = image_text.load_image_text_model("google/siglip-base-patch16-224")
qe = image_text.get_query_embedding(model, query)

scores = image_text.compute_cosine_similarity_scores(mask_text_embeddings, qe)


# Create a plotter object
plotter = pv.Plotter()
plotter.add_camera_orientation_widget()

# Add the point cloud to the plotter
plotter.add_points(point_cloud, scalars='RGB', rgb=True)

a = np.random.rand(100, 3)
ma = pv.PolyData(a)
actor = plotter.add_mesh(ma, color='white', render_points_as_spheres=True, point_size=10)


def toggle_vis(flag):
    actor.SetVisibility(flag)


plotter.add_checkbox_button_widget(toggle_vis, value=False)

# Display the plot
plotter.show()

# https://docs.pyvista.org/version/stable/examples/02-plot/color_cycler#sphx-glr-examples-02-plot-color-cycler-py
