import os
import sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

try:
    from src.preprocess import normalize_nonzero_zscore, get_bounding_box_3d
except ModuleNotFoundError:
    from preprocess import normalize_nonzero_zscore, get_bounding_box_3d


class ACDCSAMMed3DDataset(Dataset):
    def __init__(self, data_dir: str, target_shape: tuple = (128, 128, 128)):
        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        
        if not self.data_dir.exists():
            print(f"⚠️ Warning: Data folder not found at:\n   {self.data_dir}")
            self.image_files = []
            return

        all_nii = list(self.data_dir.rglob("*.nii")) + list(self.data_dir.rglob("*.nii.gz"))
        
        self.image_files = sorted([
            f for f in all_nii 
            if "_gt" not in f.name 
            and "_4d" not in f.name 
            and not f.name.startswith(".")
        ])

        if len(self.image_files) == 0:
            print(f"⚠️ Warning: No MRI scans found at:\n   {self.data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        if img_path.name.endswith(".nii.gz"):
            gt_name = img_path.name.replace(".nii.gz", "_gt.nii.gz")
        else:
            gt_name = img_path.name.replace(".nii", "_gt.nii")
            
        gt_path = img_path.parent / gt_name

        if not gt_path.exists():
            raise FileNotFoundError(f"❌ Corresponding Ground Truth file not found:\n   {gt_path}")

        img_obj = nib.load(img_path)
        gt_obj = nib.load(gt_path)

        img_data = img_obj.get_fdata().astype(np.float32)
        gt_data = gt_obj.get_fdata().astype(np.float32) # تغییر به float32 برای Interpolate

        img_norm = normalize_nonzero_zscore(img_data)

        # آماده‌سازی ابعاد برای تابع Resize (نیاز به فرمت 1x1xHxWxD دارد)
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_data).unsqueeze(0).unsqueeze(0)

        # تغییر اندازه (Resize) به 128x128x128
        img_resized = F.interpolate(img_tensor, size=self.target_shape, mode='trilinear', align_corners=False)
        gt_resized = F.interpolate(gt_tensor, size=self.target_shape, mode='nearest')

        # حذف ابعاد اضافی
        img_final = img_resized.squeeze(0)  # خروجی: (1, 128, 128, 128)
        gt_final = gt_resized.squeeze(0).long()  # خروجی: (1, 128, 128, 128)

        # محاسبه Bounding Box از روی لیبل جدید
        gt_numpy = gt_final.squeeze(0).numpy()
        bbox_3d = get_bounding_box_3d(gt_numpy, pad=2)

        if bbox_3d is None:
            bbox_3d = [0, 0, 0, 10, 10, 10]

        bbox_tensor = torch.tensor(bbox_3d, dtype=torch.float32)

        return {
            "image": img_final,
            "label": gt_final,
            "bbox": bbox_tensor,
            "filename": img_path.name
        }

if __name__ == "__main__":
    print("✅ Module src/dataset.py loaded successfully.")