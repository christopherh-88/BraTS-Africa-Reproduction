"""
Convert TCGA-LGG MRI Segmentation dataset to nnU-Net v2 2D format.

Kaggle dataset: mateuszbuda/lgg-mri-segmentation
Download locally with:
    kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
    unzip lgg-mri-segmentation.zip -d data/raw/

Input structure (inside the zip):
    kaggle_3m/
        TCGA_CS_4941_19960909/
            TCGA_CS_4941_19960909_1.tif      <- FLAIR slice
            TCGA_CS_4941_19960909_1_mask.tif <- binary mask
            TCGA_CS_4941_19960909_2.tif
            TCGA_CS_4941_19960909_2_mask.tif
            ...
        ...

Output structure (nnU-Net v2 2D):
    $nnUNet_raw/Dataset001_TCGALGG/
        imagesTr/
            TCGA_CS_4941_19960909_001_0000.nii.gz  <- FLAIR channel
            ...
        labelsTr/
            TCGA_CS_4941_19960909_001.nii.gz
            ...
        dataset.json

Each 2D slice becomes a separate nnU-Net case.
Total slices: ~3929 (1373 with tumor, 2556 without).

Usage:
    # All slices (recommended — nnU-Net handles class imbalance internally)
    python scripts/convert_tcgalgg_to_nnunet.py --input data/raw/lgg-mri-segmentation

    # Tumor-containing slices only
    python scripts/convert_tcgalgg_to_nnunet.py --input data/raw/lgg-mri-segmentation --tumor-only
"""

import argparse
import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


DATASET_JSON_TEMPLATE = {
    "channel_names": {"0": "FLAIR"},
    "labels": {"background": 0, "tumor": 1},
    "numTraining": 0,
    "file_ending": ".nii.gz",
    "name": "Dataset001_TCGALGG",
    "description": (
        "TCGA-LGG MRI Segmentation — 2D FLAIR slices from 110 lower-grade glioma patients "
        "(5 TCIA institutions). Source: mateuszbuda/lgg-mri-segmentation on Kaggle."
    ),
    "reference": "https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation",
    "licence": "CC0 1.0",
    "overwrite_image_reader_writer": "SimpleITKIO",
}


def slice_to_nifti(tif_path: Path, is_mask: bool = False) -> nib.Nifti1Image:
    """
    Load a 2D TIFF slice and return a single-slice (H, W, 1) NIfTI image.

    Images are RGB in the source files but MRI intensity is identical across channels;
    we extract the first channel. Masks are thresholded to binary {0, 1}.
    """
    arr = np.array(Image.open(tif_path))

    if arr.ndim == 3:
        arr = arr[:, :, 0]  # RGB -> single grayscale channel

    if is_mask:
        arr = (arr > 0).astype(np.uint8)
    else:
        arr = arr.astype(np.float32)

    # NIfTI convention: (H, W, Z) with Z=1 for a 2D-native case
    return nib.Nifti1Image(arr[:, :, np.newaxis], affine=np.eye(4))


def main():
    parser = argparse.ArgumentParser(
        description="Convert TCGA-LGG dataset to nnU-Net v2 2D format"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to lgg-mri-segmentation root (contains kaggle_3m/ subdir)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: $nnUNet_raw/Dataset001_TCGALGG)",
    )
    parser.add_argument(
        "--tumor-only",
        action="store_true",
        help="Only include slices with at least one tumor pixel",
    )
    args = parser.parse_args()

    input_root = Path(args.input).expanduser().resolve()
    kaggle_3m = input_root / "kaggle_3m"
    if not kaggle_3m.exists():
        kaggle_3m = input_root  # input_dir might already point to kaggle_3m/

    if not kaggle_3m.exists():
        raise FileNotFoundError(f"kaggle_3m directory not found under {input_root}")

    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        nnunet_raw = os.environ.get("nnUNet_raw")
        if not nnunet_raw:
            raise EnvironmentError(
                "nnUNet_raw is not set. Run: source setup/setup_env.sh"
            )
        output_dir = Path(nnunet_raw) / "Dataset001_TCGALGG"

    images_out = output_dir / "imagesTr"
    labels_out = output_dir / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(p for p in kaggle_3m.iterdir() if p.is_dir())
    if not patient_dirs:
        raise FileNotFoundError(f"No patient directories found in {kaggle_3m}")

    print(f"Found {len(patient_dirs)} patients in {kaggle_3m}")
    if args.tumor_only:
        print("Mode: tumor-containing slices only")
    else:
        print("Mode: all slices (tumor + non-tumor)")

    case_ids = []
    skipped_no_mask = 0
    skipped_no_tumor = 0

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        slice_files = sorted(
            f for f in patient_dir.glob("*.tif") if not f.stem.endswith("_mask")
        )

        for slice_file in slice_files:
            mask_file = slice_file.parent / f"{slice_file.stem}_mask.tif"
            if not mask_file.exists():
                print(f"  WARNING: no mask for {slice_file.name} — skipping")
                skipped_no_mask += 1
                continue

            if args.tumor_only:
                mask_arr = np.array(Image.open(mask_file))
                if mask_arr.max() == 0:
                    skipped_no_tumor += 1
                    continue

            m = re.search(r"_(\d+)$", slice_file.stem)
            slice_num = int(m.group(1)) if m else 0
            case_id = f"{patient_id}_{slice_num:03d}"

            nib.save(slice_to_nifti(slice_file, is_mask=False), images_out / f"{case_id}_0000.nii.gz")
            nib.save(slice_to_nifti(mask_file, is_mask=True), labels_out / f"{case_id}.nii.gz")
            case_ids.append(case_id)

    dataset_json = DATASET_JSON_TEMPLATE.copy()
    dataset_json["numTraining"] = len(case_ids)
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\nWrote {len(case_ids)} cases to {output_dir}")
    if skipped_no_mask:
        print(f"  Skipped (no mask found): {skipped_no_mask}")
    if skipped_no_tumor:
        print(f"  Skipped (no tumor, --tumor-only): {skipped_no_tumor}")
    print("\nNext step:")
    print("  source setup/setup_env.sh")
    print("  nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")


if __name__ == "__main__":
    main()
