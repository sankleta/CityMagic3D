#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
set -e

# RUN OPENMASK3D PHASE 1 FOR A SINGLE SCENE
# This script computes class agnostic masks and saves them

# --------
# NOTE: SET THESE PARAMETERS BASED ON YOUR SCENE!
# data paths
SCENE_DIR="$(pwd)/resources/scene_example"
SCENE_PLY_PATH="${SCENE_DIR}/scene_example.ply"
# model ckpt paths
MASK_MODULE_CKPT_PATH="$(pwd)/resources/scannet200_model.ckpt"
# output directories to save masks and mask features
EXPERIMENT_NAME="test_1"
OUTPUT_DIRECTORY="$(pwd)/output"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${TIMESTAMP}-${EXPERIMENT_NAME}"
SAVE_VISUALIZATIONS=true #if set to true, saves pyviz3d visualizations
# gpu optimization
OPTIMIZE_GPU_USAGE=false

# Set HYDRA_FULL_ERROR=1 for debugging
HYDRA_FULL_ERROR=1

cd openmask3d

# 1. Compute class agnostic masks and save them
echo "[INFO] Extracting class agnostic masks..."
python class_agnostic_mask_computation/get_masks_single_scene.py \
general.experiment_name=${EXPERIMENT_NAME} \
general.checkpoint=${MASK_MODULE_CKPT_PATH} \
general.train_mode=false \
data.test_mode=test \
model.num_queries=120 \
general.use_dbscan=true \
general.dbscan_eps=0.95 \
general.save_visualizations=${SAVE_VISUALIZATIONS} \
general.scene_path=${SCENE_PLY_PATH} \
general.mask_save_dir="${OUTPUT_FOLDER_DIRECTORY}" \
hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/class_agnostic_mask_computation" 
echo "[INFO] Mask computation done!"

# get the path of the saved masks
MASK_FILE_BASE=$(echo $SCENE_PLY_PATH | sed 's:.*/::')
MASK_FILE_NAME=${MASK_FILE_BASE/.ply/_masks.pt}
SCENE_MASK_PATH="${OUTPUT_FOLDER_DIRECTORY}/${MASK_FILE_NAME}"
echo "[INFO] Masks saved to ${SCENE_MASK_PATH}."

