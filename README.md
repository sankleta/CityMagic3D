# CityMagic3D

## Setup

### Python environment

Clone the repository, create conda environment and install the required packages using
```
conda env create -f conda_env.yml
```
(or alternatively, create a python venv and install the packages using pip from the `requirements.txt` file).

You may run the demo using our precomputed files and skipping the burden of calculations. See the Demo section below.

### Prepare the data and checkpoints. 

We used the [STPLS3D](https://www.stpls3d.com/) dataset's real world part.

Put the right paths into `config.yaml` files.

Please note the followings:
* The **point cloud** should be provided as a `.ply` file and the points are expected to be in the z-up right-handed coordinate system.
* The **camera intrinsics** and **camera poses** should be provided in a BlocksExchange `.xml` file.
* The **RGB images** should be `.png`, `.jpg`, `.jpeg` format.

Checkpoints for SAM (see [here](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) ) and a vision-text coembedding model, like CLIP are also needed.


## Preprocessing

Generate mes out of the point cloud using MeshLab using [this](https://blogs.gre.ac.uk/designsupport/3d-realisation/laser-scanning/meshlab-point-cloud-to-mesh/) instruction. 

You may use the one we've made for RA scene from [here](https://drive.google.com/file/d/1_hCSRk_LK7WxqdR_fLoVN7nH2Fsjv-ge/view?usp=sharing) 


## Generate masks from images

Prerequisites: a mesh of the scene (see above), a set of images with the camera intrinsics (get them from [here](https://drive.google.com/drive/folders/1nOtyygYrVCu0puRuJTFqN9-gv6kA2E4J) ).

To run in the root directory, by:
```
python -m instance_masks_from_images.main
```

By default, it uses the `./instance_masks_from_images/config.yaml` as config file, which can be overriden by 
```
python -m instance_masks_from_images.main --config-name=config_v2.yaml
```

This script first extracts and saves the class-agnostic masks, and then computes the per-mask features. Masks and mask-features are saved into the directory specified by the user at the beginning of [this script](run_openmask3d_single_scene.sh). In particular, the output has the following structure.
```
OUTPUT_FOLDER_DIRECTORY
      └── date-time-experiment_name                           <- folder with the output of a specific experiment
             ├── crops                                        <- folder with crops (if SAVE_CROPS=true)
             ├── hydra_outputs                                <- folder with outputs from hydra (config.yaml files are useful)
             ├── scene_example_masks.pt                       <- class-agnostic instance masks - dim. (num_points, num_masks) indicating the masks in which a given point is included
             └── scene_example_openmask3d_features.npy        <- per-mask features for each object instance - dim. (num_masks, num_features), the mask-feature vecture for each instance mask. 
```

## Assemble masks

This phase can be run by
```
python -m merge_masks.main
```

Don't forget to set the parameters in `merge_masks/cofig.yaml`. A different config file can be set as an argument of the above command as for the first phase.
As `input_dir` parameter, an output directory of the first phase is reequired.
The point cloud and the camera info files should be the same as in the first phase.


## Demo
The demo is a console application.
[Here](https://drive.google.com/file/d/1_hCSRk_LK7WxqdR_fLoVN7nH2Fsjv-ge/view?usp=sharing) you can download the archive with all the files you need for the demo or use the files generated during the "Assemble masks" step.

Unzip the files and fix the paths in demo/config.yaml according to your environment. Run demo/main.py. It takes some time to load the data and get ready for querying.
Once it's ready, you'll get the message ```Data loaded. Enter your query. Query (type 'exit' to quit): ```. 

Type your query and press Enter. 

The window will pop up with the visualization showing the top 20 instance masks with the highest cosine similarity score colored into different colors. They might be small. 

You may adjust the number of top masks in demo/config.yaml.

Use [hotkeys](https://docs.pyvista.org/version/stable/api/plotting/plotting.html) to navigate the scene.

Once the window with the visualization is closed, you'll get again the prompt to enter the query.

