# ACDC-SAMMed3D

> 3D medical image segmentation research using SAM-Med3D and the ACDC cardiac imaging dataset.

## Overview

ACDC-SAMMed3D is an end-to-end research framework for exploring 3D foundation-model approaches to cardiac medical-image segmentation. It uses volumetric data rather than independent 2D slices and investigates prompt-driven segmentation with a pretrained SAM-Med3D model.

## Key Components

- Full 3D medical-image pipeline
- Dynamic 3D bounding-box prompts from ground-truth masks
- Combined Dice and Cross-Entropy loss
- Modular dataset, preprocessing, model, loss, and metric components
- Training and evaluation scripts

## Architecture

```text
ACDC Volumes
     ↓
Preprocessing / Normalization
     ↓
3D Prompt Generation
     ↓
SAM-Med3D
     ↓
Segmentation Mask
     ↓
Dice / IoU Evaluation
```

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

The framework is designed to evaluate segmentation with:

- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Visual prediction comparisons

No new benchmark number is claimed by this README unless it is backed by an executed experiment.

## Development Status

**Research project / active refinement.**

The current repository provides a structured implementation foundation. Reproducibility, dependency pinning, experiment configuration, and broader validation can be strengthened as development continues.

## Future Work

- Reproducible configuration management
- Automated tests
- Experiment tracking
- Robust checkpoint management
- Additional segmentation metrics
- Cross-validation / robustness analysis
- Better documentation of preprocessing and model dependencies

## Technology

Python • PyTorch • NumPy • NiBabel • 3D Medical Imaging • SAM-Med3D

## Acknowledgments

This work builds on the SAM-Med3D ecosystem and the ACDC dataset.

## Author

Mohammad Mahdi Shafighi — M.Sc. Artificial Intelligence
