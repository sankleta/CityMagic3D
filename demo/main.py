import numpy as np
import pyvista as pv

from instance_masks_from_images import clip

point_cloud = pv.read(r"C:\Users\sankl\Downloads\WMSC\WMSC_points.ply")

query = "vegetation"
clip.load_clip("google/siglip-base-patch16-224")
qe = clip.get_query_embedding(clip, query)



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
