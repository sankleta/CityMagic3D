# CityMagic3D


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





---
---
---
## Setup for OpenMask3d 🛠
Clone the repository, create conda environment and install the required packages as follows:
```bash
conda create --name=openmask3d python=3.8.5 # create new virtual environment
conda activate openmask3d # activate it
bash openmask3d/install_requirements.sh  # install requirements
pip install -e .  # install current repository in editable mode
```
Note: If you encounter any issues in the `bash install_requirements.sh` step, we recommend you to run the commands in that script one-by-one, especially for performing the MinkowskiEngine installation manually. 

---


## Run OpenMask3D on a single scene 🛋
In this section we provide some information about how to run OpenMask3D on a single scene. In particular, we divide this section into four parts:
1. Download **checkpoints**
2. Check the format of **scene's data**
3. Set-up **configurations** 
4. **Run** OpenMask3D 

### Step 1: Download the checkpoints 📍
Create a folder `resources` in the main directory of the repository. Then, add to this folder the checkpoints for:
* **Mask module network**: use [this link](https://drive.google.com/file/d/1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B/view?usp=sharing) (model trained on ScanNet200 training set) for evaluating on **ScanNet validation scenes**, or [this link](https://drive.google.com/file/d/1rD2Uvbsi89X4lSkont_jUTT7X9iaox9y/view?usp=share_link) for running the model on an **arbitrary scene**.
* **Segment Anything Model** (in our case we used ViT-H): use this [link](https://drive.google.com/file/d/1WHi0hBi0iqMZfk8l3rDXLrW4lEEgHm_y/view?usp=sharing) or the [official repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).


### Step 2: Check the folder structure of the data for your scene 🛢
In order to run OpenMask3D you need to have access to the point cloud of the scene as well to the posed RGB-D frames.

We recommend creating a folder `scene_example` inside the `resources` folder where the data is saved with the following structure ([here](https://drive.google.com/file/d/1UOwBZMCrTMg-_MFwmYkKOrex1YS6Nw-i/view?usp=sharing) we provide a scene as an example). 
```
scene_example
      ├── pose                            <- folder with camera poses
      │      ├── 0.txt 
      │      ├── 1.txt 
      │      └── ...  
      ├── color                           <- folder with RGB images
      │      ├── 0.jpg (or .png/.jpeg)
      │      ├── 1.jpg (or .png/.jpeg)
      │      └── ...  
      ├── depth                           <- folder with depth images
      │      ├── 0.png (or .jpg/.jpeg)
      │      ├── 1.png (or .jpg/.jpeg)
      │      └── ...  
      ├── intrinsic                 
      │      └── intrinsic_color.txt       <- camera intrinsics
      └── scene_example.ply                <- point cloud of the scene
```

Please note the followings:
* The **point cloud** should be provided as a `.ply` file and the points are expected to be in the z-up right-handed coordinate system.
* The **camera intrinsics** and **camera poses** should be provided in a `.txt` file, containing a 4x4 matrix.
* The **RGB images** and the **depths** can be either in `.png`, `.jpg`, `.jpeg` format; the used format should be specified as explained in **Step 3**.
* The **RGB images** and their corresponding **depths** and **camera poses** should be named as `{FRAME_ID}.extension`, without zero padding for the frame ID, starting from index 0.


### Step 3: Set-up the paths to data and to output folders 🛤
Before running OpenMask3D make sure to fill all the required parameters in [this script](run_openmask3d_single_scene.sh). In particular, if you have followed the structure provided in Step 2, you should adapt only the following fields:
* `SCENE_DIR`: directory to `scene_example`
* `SCENE_INTRINSIC_RESOLUTION`: resolution on which intrinsics are computed
* `IMG_EXTENSION`: extension of RGB pictures. Either `.png`, `.jpg`, `.jpeg`
* `DEPTH_EXTENSION`: extension of depth pictures. Either `.png`, `.jpg`, `.jpeg`
* `DEPTH_SCALE`: factor by which the depth of the sensor should be divided to obtain a measure in terms of meters. It should be set to 1000 for ScanNet depth images and to 6553.5 for Replica depth images. You should set this value based on the scale of your depth maps.
* `MASK_MODULE_CKPT_PATH`: path to the mask module network checkpoint
* `SAM_CKPT_PATH`: path to the Segment Anything Model (SAM) checkpoint
* `OUTPUT_FOLDER_DIRECTORY`: path to the folder in which you wish to save the outputs
* `SAVE_VISUALIZATIONS`: set to true if you wish to save the visualizations of the class-agnostic masks
* `SAVE_CROPS`: set to true if you wish to save the 2D crops of the masks from which the CLIP features are extracted. It can be helpful for debugging and for a qualitative evaluation of the quality of the masks.
* `OPTIMIZE_GPU_USAGE`: set to true if you have some memory constraints and wish to minimize GPU memory footprint. Please note that this version is slower compared to the our default version.


### Step 4: Run OpenMask3D 🚀
Now you can run OpenMask3D by using the following command.
```bash
bash run_openmask3d_single_scene.sh
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


Note: For the ScanNet validation, we use available segments on ScanNet and obtain more robust and less noisy masks compared to directly running the mask predictor on the point cloud. Therefore, the results we obtain for a single scene from ScanNet directly using the point cloud can be different then the masks obtained during the overall ScanNet evaluation described in the section below. 

