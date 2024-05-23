from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3, html

import pyvista as pv
from pyvista.trame.ui import plotter_ui

import numpy as np

from demo.utils import show_top_n_instances, show_heatmap, clean
from instance_masks_from_images import image_text

# -----------------------------------------------------------------------------
# Trame initialization
# -----------------------------------------------------------------------------

pv.OFF_SCREEN = True

server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

TEXT_EMBEDDING_SIZE = 512

MESH = pv.read(r"C:\Users\sankl\Downloads\RA\RA_1M.ply")
LOADED_MASKS = np.load(r'C:\Users\sankl\Downloads\2024-05-21_22-26-11\merged_masks.npz')
LOADED_TEXT_EMBEDDINGS = np.load(r'C:\Users\sankl\Downloads\2024-05-21_22-26-11\merged_embeddings__max.npz')

mask_embeddings = {key: LOADED_MASKS[key] for key in LOADED_MASKS}

mask_text_embeddings = {key: LOADED_TEXT_EMBEDDINGS[key].reshape(TEXT_EMBEDDING_SIZE, ) for key in
                        LOADED_TEXT_EMBEDDINGS}

clean(mask_embeddings, mask_text_embeddings)

model = image_text.load_image_text_model("openai/clip-vit-base-patch32")

state.trame__title = "CityMagic3D"
ctrl.on_server_ready.add(ctrl.view_update)


def on_submit():
    input_value = state.textbox_value
    query_embeddings = image_text.get_query_embedding(model, input_value)
    scores = image_text.compute_cosine_similarity_scores(mask_text_embeddings, query_embeddings)
    plotter.subplot(0, 0)

    # Add the mesh to the plotter
    show_heatmap(plotter, MESH, scores, mask_embeddings)

    plotter.subplot(0, 1)
    # Add masks as points on top
    show_top_n_instances(plotter, MESH, scores, mask_embeddings, 5)
    plotter.update()
    #state.display_text = f"Updated titles to: {input_value}"

# Bind the callback to the server
ctrl.on_submit = on_submit

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

# # Create a plotter object
plotter = pv.Plotter(shape=(1, 2))
plotter.subplot(0, 0)

# Add the mesh to the plotter
plotter.add_mesh(MESH, scalars='RGBA', rgb=True)
# actor = show_heatmap(plotter, MESH, scores, mask_embeddings)
#
plotter.subplot(0, 1)
plotter.add_mesh(MESH, scalars='RGBA', rgb=True)
# Add masks as points on top
# show_top_n_instances(plotter, MESH, scores, mask_embeddings, 17)
#
plotter.link_views()

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------


with SinglePageLayout(server) as layout:
    with layout.toolbar:
        vuetify3.VSwitch(
            v_model="$vuetify3.theme.dark",
            hide_details=True,
            dense=True,
        )

    with layout.content:
        view = plotter_ui(plotter)
        ctrl.view_update = view.update

    with layout.footer:
        with vuetify3.VRow():
            with vuetify3.VCol(cols=8):
                vuetify3.VTextField(
                    v_model=("textbox_value", ""),
                    label="Enter your query",
                    placeholder="Type here",
                )
            with vuetify3.VCol(cols=4):
                vuetify3.VBtn(
                    "Submit",
                    click=server.controller.on_submit,
                    color="primary",
                )

    #layout.footer.hide()

state.textbox_value = ""
state.display_text = "No query"

if __name__ == "__main__":
    server.start()
