# BBBC041 Malaria Cell Imaging - Fast Proof Baseline

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-optional-76B900?logo=nvidia&logoColor=white) ![Tests](https://img.shields.io/badge/tests-pytest-blue)

This repo is a reproducible baseline for bioimage classification using the Broad Bioimage Benchmark Collection (BBBC041). It turns object annotations into per-cell crops and trains a transfer-learning CNN with clear metrics, embeddings, and Grad-CAMs.

Project highlights: Transfer-learning ResNet-34 on BBBC041 with reproducible splits, interpretability (UMAP + Grad-CAM), and external validation on NIH malaria images.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you want GPU wheels for CUDA, install torch/torchvision with the official index URL, for example:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

RTX 50-series (sm_120) requires nightly CUDA 12.8 builds:

```bash
pip install --pre --upgrade torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Prepare data and build the index:

```bash
python -m bioimage.cli prepare --label-mode binary
```

Train a baseline (GPU recommended):

```bash
python -m bioimage.cli train --label-mode binary --epochs 5 --batch-size 128 --freeze-backbone
```

Evaluate and export metrics + confusion matrix:

```bash
python -m bioimage.cli eval --model-path models/bbbc041_binary_best.pt
```

Generate UMAP embeddings + Grad-CAM figures:

```bash
python -m bioimage.cli make-figures --model-path models/bbbc041_binary_best.pt
```

Outputs land in `reports/`.

Run tests:

```bash
pytest
```

## External Validation (NIH Malaria Cell Images)

This dataset is from the U.S. National Library of Medicine and is independent of BBBC041.

Download and evaluate:

```bash
python -m bioimage.cli prepare-external
python -m bioimage.cli eval-external --model-path models/bbbc041_binary_best.pt
```

Outputs:
- `reports/external_nih_metrics.csv`
- `reports/external_nih_confusion_matrix.png`

## Dataset

- **BBBC041** (malaria thin blood smears) from the Broad Bioimage Benchmark Collection.
- Annotations are provided as bounding boxes in `training.json`.
- This pipeline converts object boxes into **cell crops** for classification.

Labeling strategy (default `binary` mode):
- **uninfected**: red blood cell
- **infected**: trophozoite, ring, schizont, gametocyte
- Dropped: `difficult` and `leukocyte`

## Preprocessing

- Crop each cell using its bounding box (+10% margin).
- Resize to 224x224 for ResNet-34.
- Normalize **per-image** (per-channel z-score).
- Augmentations (train): random resized crop, flips, rotation, stronger intensity jitter, blur, random erasing.
- All crops are converted to RGB (grayscale replicated if needed).

## Model

- **ResNet-34** pre-trained on ImageNet (frozen backbone, linear probe).
- Classifier-only training to reduce overfitting.
- Balanced sampling for class imbalance (no class weighting when sampling is enabled).

## Metrics & Artifacts

The pipeline reports:
- Accuracy
- Macro F1
- Balanced accuracy

Artifacts saved in `reports/`:
- `metrics.csv` (val/test table)
- `confusion_matrix.png`
- `umap_embeddings.png`
- `gradcam_correct.png`, `gradcam_incorrect.png`

Key figures (frozen-backbone baseline):
- `reports/confusion_matrix_best.png`
- `reports/umap_embeddings.png`
- `reports/gradcam_correct.png`
- `reports/gradcam_incorrect.png`
- `reports/external_nih_confusion_matrix_best.png`

## Methods & Results

**Repro Commands**
- Train (frozen backbone, recommended for external generalization): `python -m bioimage.cli train --label-mode binary --epochs 5 --batch-size 128 --freeze-backbone`
- Eval (internal): `python -m bioimage.cli eval --model-path models/bbbc041_binary_best.pt`
- Eval (external): `python -m bioimage.cli eval-external --model-path models/bbbc041_binary_best.pt`

**Problem**
- Classify malaria-infected vs uninfected red blood cells from BBBC041 cell crops.

**Data**
- BBBC041 object annotations converted to cell crops.
- Train/val/test split by image with fixed seed to avoid leakage.

**Model**
- ResNet-34 transfer learning, frozen backbone + linear classifier.

**Training Setup**
- Optimizer: AdamW
- Epochs: 5
- Batch size: 128
- Normalization: per-image z-score
- Augmentations: random resized crop (0.75-1.0), flips, rotation (15 deg), color jitter, blur, random erasing
- Class imbalance: weighted sampler (no class weighting in loss)
- Regularization: dropout (0.4), label smoothing (0.1)

**Metrics (val/test)**
- Accuracy: 0.9552 / 0.9536
- Macro F1: 0.7305 / 0.7286
- Balanced accuracy: 0.9374 / 0.9305

Interpretation: The confusion matrix shows most errors in the infected minority class, which matches the class imbalance. UMAP shows partial separation with overlap, and Grad-CAM highlights localized inclusions inside cells rather than background.

**External Validation (NIH Malaria Cell Images)**
- Accuracy: 0.7142
- Macro F1: 0.6925
- Balanced accuracy: 0.7142

Interpretation: Performance drops on the external dataset, which is expected due to acquisition and staining differences (domain shift). This still indicates useful cross-dataset transfer and highlights the need for domain adaptation or mixed-source training in production.

**What Worked**
- Transfer learning converged quickly and produced clear separation in UMAP.

**Next Steps for Production**
- Stratify by donor/slide or plate to avoid leakage.
- Add stain normalization and explicit illumination correction.
- Explore self-supervised pretraining + multi-stage parasite classification.

## Repo Structure

```
src/bioimage/        # dataset, model, training, eval, figures
reports/             # metrics tables and figures
```

## Notes

- Default run uses the full dataset. For a fast smoke test, pass `--max-samples 2000` to train/eval.
- All splits are reproducible via seed.
- Raw data and model checkpoints are not committed to keep the repo lightweight.
