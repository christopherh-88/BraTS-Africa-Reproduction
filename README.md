# BraTS-Africa Reproduction

Reproducing and extending the nnU-Net **2D** baseline on the [TCGA-LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (110 lower-grade glioma patients, 5 TCIA institutions), with extensions for responsible AI and low-resource clinical deployment.

## Project Structure

```
BraTS-Africa-Reproduction/
├── data/               # Local data dir (gitignored — do not commit scans)
├── notebooks/
│   └── kaggle_train_fold.ipynb  # Full Kaggle training notebook (Phases 1–2)
├── scripts/
│   ├── convert_tcgalgg_to_nnunet.py  # TIFF slices → nnU-Net 2D format
│   ├── aggregate_results.py           # Average metrics across folds
│   ├── quantize_model.py              # Phase 3: INT8/FP16 quantization
│   └── robustness_test.py             # Phase 3: noise/degradation tests
├── results/            # Aggregated metrics, plots, tables
├── setup/
│   └── setup_env.sh    # Source this to set nnU-Net environment variables
├── environment.yml     # Conda environment (local/CPU development)
└── README.md
```

## Roadmap

| Phase | Deadline | Status |
|-------|----------|--------|
| 0+1: Setup + Data Prep | 2026-05-23 | — |
| 2: Reproduction (3-fold CV) | 2026-05-27 | — |
| 3: Extensions | 2026-06-02 | — |
| 4: Writing & Submission | 2026-06-04 | — |

---

## Phase 0: Environment Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate brats
```

> **Kaggle / CUDA:** `environment.yml` installs CPU torch for local dev.
> On Kaggle (GPU), the notebook installs the right packages automatically.

### 2. Configure nnU-Net paths

Edit the paths in `setup/setup_env.sh` to match your local directories, then source it before any `nnunetv2` commands:

```bash
source setup/setup_env.sh
```

To make this permanent across sessions, add the exports to `~/.zshrc` or `~/.bashrc`.

| Variable | Purpose |
|---|---|
| `nnUNet_raw` | Raw datasets (e.g. `Dataset001_TCGALGG/`) |
| `nnUNet_preprocessed` | Output of `nnUNetv2_plan_and_preprocess` |
| `nnUNet_results` | Model checkpoints from `nnUNetv2_train` |

### 3. Verify installation

```bash
nnUNetv2_train --help
python -c "import nnunetv2; print(nnunetv2.__version__)"
```

---

## Phase 1: Data Preparation

### Download the dataset

```bash
# Install the Kaggle CLI if needed: pip install kaggle
# Place your kaggle.json API token at ~/.kaggle/kaggle.json
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
unzip lgg-mri-segmentation.zip -d data/raw/
```

The zip extracts to `data/raw/lgg-mri-segmentation/kaggle_3m/` with one folder per patient:

```
kaggle_3m/
  TCGA_CS_4941_19960909/
    TCGA_CS_4941_19960909_1.tif       <- 2D FLAIR slice
    TCGA_CS_4941_19960909_1_mask.tif  <- binary tumor mask
    ...
```

### Convert to nnU-Net 2D format

```bash
source setup/setup_env.sh

python scripts/convert_tcgalgg_to_nnunet.py \
    --input data/raw/lgg-mri-segmentation
```

This writes ~3929 cases (1 slice = 1 case) to `$nnUNet_raw/Dataset001_TCGALGG/`:

```
$nnUNet_raw/Dataset001_TCGALGG/
├── imagesTr/
│   ├── TCGA_CS_4941_19960909_001_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── TCGA_CS_4941_19960909_001.nii.gz
│   └── ...
└── dataset.json
```

### Run fingerprinting and preprocessing

```bash
nnUNetv2_plan_and_preprocess -d 1 -c 2d --verify_dataset_integrity
```

---

## Phase 2: Training (3-Fold CV on Kaggle)

Open `notebooks/kaggle_train_fold.ipynb` on Kaggle:
1. Add `mateuszbuda/lgg-mri-segmentation` as a Kaggle input dataset.
2. Set `FOLD` to your assigned fold.
3. Run All — the notebook handles conversion, preprocessing, and training end-to-end.

| Person | Fold | Command |
|--------|------|---------|
| Person A | 0 | `nnUNetv2_train Dataset001_TCGALGG 2d 0 --npz` |
| Person B | 1 | `nnUNetv2_train Dataset001_TCGALGG 2d 1 --npz` |
| Person C | 2 | `nnUNetv2_train Dataset001_TCGALGG 2d 2 --npz` |

**Key metrics to capture:** Dice (target ~0.82–0.92), IoU, Hausdorff Distance (HD95).  
Also document: inference time, peak GPU memory, training curves.

After all three folds finish, one person aggregates results:

```bash
python scripts/aggregate_results.py
```

---

## Phase 3: Extensions (Low-Resource Focus)

- **Quantization:** INT8 / FP16 inference (`scripts/quantize_model.py`)
- **Knowledge distillation:** smaller student model from full nnU-Net
- **Robustness:** noise injection, resolution degradation, missing slices (`scripts/robustness_test.py`)
- **Cross-institution generalization:** per-TCIA-center performance (institution code embedded in patient ID, e.g. `TCGA_CS_`, `TCGA_DU_`, ...)
- **Fairness:** performance by tumor size and genomic cluster; equity analysis for quantized vs. full model

---

## Phase 4: Paper

Introduction (health equity) → Reproduction details → Extensions & Results → Limitations.
