import os
import sys
from pathlib import Path

# اضافه کردن مسیر ریشه پروژه به sys.path برای جلوگیری از ModuleNotFoundError
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

# وارد کردن توابع پیش‌پردازش از src.preprocess
try:
    from src.preprocess import normalize_nonzero_zscore, get_bounding_box_3d
except ModuleNotFoundError:
    from preprocess import normalize_nonzero_zscore, get_bounding_box_3d


class ACDCSAMMed3DDataset(Dataset):
    """
    Dataset class for loading ACDC MRI scans and generating 3D Bounding Box / Point prompts
    for SAM-Med3D fine-tuning based on paper specification (128x128x128 patches).
    """
    def __init__(self, data_dir: str, target_shape: tuple = (128, 128, 128)):
        """
        Args:
            data_dir: مسیر پوشه داده‌ها (شامل فایل‌های image و label)
            target_shape: ابعاد مکعبی ورودی مدل SAM-Med3D (مطابق مقاله)
        """
        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        
        # پیدا کردن فایل‌های تصویر اصلی (به‌جز فایل‌های GT)
        self.image_files = sorted([
            f for f in self.data_dir.glob("*.nii.gz") 
            if "_gt" not in f.name
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        gt_path = Path(str(img_path).replace(".nii.gz", "_gt.nii.gz"))

        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        # ۱. بارگذاری داده‌های NIfTI
        img_obj = nib.load(img_path)
        gt_obj = nib.load(gt_path)

        img_data = img_obj.get_fdata().astype(np.float32)
        gt_data = gt_obj.get_fdata().astype(np.int64)

        # ۲. پیش‌پردازش و نرمال‌سازی Z-score
        img_norm = normalize_nonzero_zscore(img_data)

        # ۳. استخراج Bounding Box ۳ بعدی به عنوان Prompt
        bbox_3d = get_bounding_box_3d(gt_data, pad=2)

        if bbox_3d is None:
            bbox_3d = [0, 0, 0, 10, 10, 10]

        # ۴. تبدیل به Tensorهای PyTorch
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(gt_data).unsqueeze(0).long()
        bbox_tensor = torch.tensor(bbox_3d, dtype=torch.float32)

        return {
            "image": img_tensor,
            "label": gt_tensor,
            "bbox": bbox_tensor,
            "filename": img_path.name
        }

if __name__ == "__main__":
    print("✅ Module src/dataset.py loaded successfully.")