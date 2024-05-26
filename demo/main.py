import numpy as np
import pyvista as pv

from demo.utils import show_top_n_instances, show_heatmap
from instance_masks_from_images import image_text

import hydra
from omegaconf import DictConfig


def load_data(cfg):
    mesh = pv.read(cfg.mesh_path)
    loaded_masks = np.load(cfg.merged_masks)
    loaded_text_embeddings = np.load(cfg.merged_embeddings)

    mask_embeddings = {key: loaded_masks[key] for key in loaded_masks}

    mask_text_embeddings = {key: loaded_text_embeddings[key].reshape(cfg.image_text_embedding_size, ) for key in
                            loaded_text_embeddings}

    model = image_text.load_image_text_model("openai/clip-vit-base-patch32")
    return mesh, mask_embeddings, mask_text_embeddings, model


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    print("Loading data...")
    mesh, mask_embeddings, mask_text_embeddings, model = load_data(cfg)
    print("Data loaded. Enter your query:")

    while True:
        query = input("Query (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        print("Processing query...")
        query_embeddings = image_text.get_query_embedding(model, query)
        scores = image_text.compute_cosine_similarity_scores(mask_text_embeddings, query_embeddings)
        print("Displaying mesh...")
        plotter = pv.Plotter()
        # plotter.subplot(0, 0)
        plotter.add_text(f"Query: '{query}'", font_size=14)

        # # Add the mesh to the plotter
        # plotter.add_mesh(MESH, scalars='RGBA', rgb=True)
        # show_heatmap(plotter, MESH, scores, mask_embeddings)
        #
        # plotter.subplot(0, 1)
        plotter.add_mesh(mesh, scalars='RGBA', rgb=True)
        # Add masks as points on top
        show_top_n_instances(plotter, mesh, scores, mask_embeddings, 20)
        plotter.show()


if __name__ == "__main__":
    main()
