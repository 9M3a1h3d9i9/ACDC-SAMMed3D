# 🧠 ACDC SAM-Med3D Fine-Tuning Framework

An end-to-end framework for fine-tuning the 3D Segment Anything Model (SAM-Med3D) on the ACDC (Automated Cardiac Diagnosis Challenge) dataset. This project implements a fully 3D architecture for medical image segmentation, leveraging pre-trained volumetric adapters.

## 🌟 Key Features
* **Full 3D Segmentation:** Utilizes SAM-Med3D instead of 2D slice-by-slice processing.
* **Optimized Prompting:** Extracts 3D bounding boxes dynamically from ground-truth masks.
* **Combined Loss Function:** Integrates Dice Loss and Cross-Entropy for optimal handling of highly imbalanced medical data.
* **Modular Architecture:** Clean, maintainable, and highly scalable codebase.

## 📂 Project Structure
```text
ACDC-SAMMed3D/
├── data/
│   ├── raw/                  # Original .nii.gz ACDC files
│   └── processed/            # Pre-processed and partitioned data
├── src/
│   ├── __init__.py
│   ├── dataset.py            # Custom 3D DataLoader
│   ├── model.py              # SAM-Med3D model loader & configurations
│   ├── losses.py             # Combined Dice & CE Loss
│   ├── metrics.py            # Dice Coefficient & IoU evaluation
│   └── preprocess.py         # Z-score normalization & BBox generation
├── ckpt/                     # Saved model checkpoints
├── results/                  # Visual predictions & evaluation output
├── train.py                  # Main training and fine-tuning loop
├── evaluate.py               # Evaluation script for generating metrics and plots
└── README.md

🚀 Getting Started
1. Prerequisites
Ensure you have PyTorch (with CUDA support) installed. Then, set up the required packages:

Bash
pip install torch torchvision torchaudio
pip install nibabel numpy matplotlib
Note: Make sure the original SAM_Med3D repository is cloned adjacent to this project to access the core model architecture and pre-trained weights.

2. Prepare the Data
Place the ACDC dataset .nii.gz files inside the data/raw/ directory. Ensure each image has its corresponding _gt.nii.gz label file.

3. Training
To start fine-tuning the model on your dataset, run:

Bash
python train.py
The script automatically saves the best-performing model to ckpt/best_sam_med3d_acdc.pth based on the Validation Dice Score.

4. Evaluation
To evaluate the model and generate 2D/3D visual comparisons, run:

Bash
python evaluate.py
Outputs will be saved in the results/ folder.

📈 Evaluation Metrics
This framework utilizes the following metrics to ensure robust medical image evaluation:

Dice Similarity Coefficient (DSC): Primary metric for measuring spatial overlap.

Intersection over Union (IoU): Secondary accuracy metric.

🤝 Acknowledgments
Based on the official SAM-Med3D implementation.

Dataset provided by ACDC (Automated Cardiac Diagnosis Challenge).


