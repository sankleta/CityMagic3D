# CityMagic3D

## Setup

Clone the repository, create conda environment and install the required packages.
Prepare the data and checkpoints.
Put the right paths into config.yaml files.

Please note the followings:
* The **point cloud** should be provided as a `.ply` file and the points are expected to be in the z-up right-handed coordinate system.
* The **camera intrinsics** and **camera poses** should be provided in a BlocksExchange `.xml` file.
* The **RGB images** should be `.png`, `.jpg`, `.jpeg` format.


## Preprocessing
Generate mes out of the point cloud using MeshLab using this instruction

## Generate masks from images

Prerequisites: a point cloud and a mesh of the same scene, and a set of images.

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

## Demo
