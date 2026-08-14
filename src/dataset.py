import os
import sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
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
            # print(f"⚠️ هشدار: پوشه داده‌ها در مسیر زیر وجود ندارد:\n   {self.data_dir}")
            print(f"⚠️ Warning: Data folder not found at:\n   {self.data_dir}")
            self.image_files = []
            return

        # جستجوی تودرتو (rglob) برای پیدا کردن تمام فایل‌های تصویر در ساب‌فولدرها
        all_nii = list(self.data_dir.rglob("*.nii")) + list(self.data_dir.rglob("*.nii.gz"))
        
        # فیلتر کردن فایل‌های GT و فایل‌های 4D
        self.image_files = sorted([
            f for f in all_nii 
            if "_gt" not in f.name 
            and "_4d" not in f.name 
            and not f.name.startswith(".")
        ])

        if len(self.image_files) == 0:
            # print(f"⚠️ هشدار: هیچ فایل اسکن MRI در مسیر زیر یافت نشد:\n   {self.data_dir}")
            print(f"⚠️ Warning: No MRI scans found at:\n   {self.data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # پیدا کردن مسیر فایل Ground Truth در همان پوشه بیمار
        if img_path.name.endswith(".nii.gz"):
            gt_name = img_path.name.replace(".nii.gz", "_gt.nii.gz")
        else:
            gt_name = img_path.name.replace(".nii", "_gt.nii")
            
        gt_path = img_path.parent / gt_name

        if not gt_path.exists():
            # raise FileNotFoundError(f"❌ فایل Ground Truth معادل پیدا نشد:\n   {gt_path}")
            raise FileNotFoundError(f"❌ Corresponding Ground Truth file not found:\n   {gt_path}")

        img_obj = nib.load(img_path)
        gt_obj = nib.load(gt_path)

        img_data = img_obj.get_fdata().astype(np.float32)
        gt_data = gt_obj.get_fdata().astype(np.int64)

        img_norm = normalize_nonzero_zscore(img_data)
        bbox_3d = get_bounding_box_3d(gt_data, pad=2)

        if bbox_3d is None:
            bbox_3d = [0, 0, 0, 10, 10, 10]

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
