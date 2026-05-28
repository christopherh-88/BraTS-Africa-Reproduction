#!/usr/bin/env bash
# Configure nnU-Net environment variables.
# Source this file (do NOT run it directly):
#   source setup/setup_env.sh
#
# nnU-Net requires three directories. Edit the paths below to match your
# local or Kaggle setup, then source this script before any nnunetv2 commands.

# --- Edit these paths ---

# Raw dataset root (contains Dataset001_TCGALGG/)
export nnUNet_raw="${HOME}/nnunet_data/nnUNet_raw"

# Preprocessed data (written by nnUNetv2_plan_and_preprocess)
export nnUNet_preprocessed="${HOME}/nnunet_data/nnUNet_preprocessed"

# Model checkpoints and results (written by nnUNetv2_train)
export nnUNet_results="${HOME}/nnunet_data/nnUNet_results"

# --- No edits needed below ---

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "nnU-Net environment configured:"
echo "  nnUNet_raw          = $nnUNet_raw"
echo "  nnUNet_preprocessed = $nnUNet_preprocessed"
echo "  nnUNet_results      = $nnUNet_results"
