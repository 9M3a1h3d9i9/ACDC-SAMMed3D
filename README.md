# ACDC-SAMMed3D

> **3D Medical Image Segmentation Research** using the ACDC cardiac dataset and SAM-Med3D-style foundation-model workflows.

## Overview

ACDC-SAMMed3D is an applied medical-AI research project focused on volumetric cardiac image segmentation. It investigates how 3D foundation-model approaches can be integrated into a reproducible segmentation pipeline instead of treating a volume as independent 2D slices.

## Research Pipeline

```text
ACDC Volumes
     ↓
Preprocessing & Normalization
     ↓
3D Prompt Generation
     ↓
SAM-Med3D-based Model
     ↓
Predicted Segmentation
     ↓
Dice / IoU / Visual Analysis
```

## Key Components

- 3D medical-image data pipeline
- Dynamic bounding-box prompt generation from masks
- Modular dataset and preprocessing components
- Segmentation loss and metric modules
- Training and evaluation entry points
- Experiment-oriented result organization

## Repository Structure

```text
ACDC-SAMMed3D/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   └── preprocess.py
├── ckpt/
├── results/
├── train.py
├── evaluate.py
└── README.md
```

## Evaluation

The intended evaluation protocol includes:

- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Qualitative prediction comparisons
- Robustness and cross-validation analysis as the project matures

**Research integrity:** this README does not claim a new benchmark number unless it is supported by an executed experiment.

## Development Status

**Research project / active refinement.** The repository has a structured implementation foundation, while reproducibility, dependency management, experiment tracking, and broader validation remain ongoing work.

## Roadmap

- [ ] Pin reproducible dependencies and model versions.
- [ ] Add automated tests for preprocessing, prompts, losses, and metrics.
- [ ] Improve checkpoint and experiment management.
- [ ] Add stronger evaluation and visualization tooling.
- [ ] Perform robustness and cross-validation studies.
- [ ] Document preprocessing and model assumptions in detail.

## Technology

`Python` · `PyTorch` · `NumPy` · `NiBabel` · `3D Medical Imaging` · `SAM-Med3D`

## Author

**Mohammad Mahdi Shafighi** — M.Sc. Artificial Intelligence