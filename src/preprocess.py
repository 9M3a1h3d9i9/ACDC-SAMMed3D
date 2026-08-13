import os
import sys

from pathlib import Path

# اضافه کردن مسیر ریشه پروژه به sys.path برای جلوگیری از ModuleNotFoundError
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import nibabel as nib

def get_bounding_box_3d(mask: np.ndarray, pad: int = 2) -> list:
    """
    استخراج Bounding Box سه‌بعدی از ماسک گراند-تروث برای استفاده به عنوان Prompt در SAMMed3D.
    خروجی: [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    non_zero = np.argwhere(mask > 0)
    if len(non_zero) == 0:
        return None
    
    min_coords = np.min(non_zero, axis=0) - pad
    max_coords = np.max(non_zero, axis=0) + pad
    
    # رعایت مرزهای حجم تصویر
    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, np.array(mask.shape) - 1)
    
    return np.concatenate([min_coords, max_coords]).tolist()

def normalize_nonzero_zscore(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    نرمال‌سازی Z-Score فقط روی وکسل‌های غیرصفر
    """
    img = img.astype(np.float32)
    mask = img > 0
    if not np.any(mask):
        return img * 0.0
    
    mean = np.mean(img[mask])
    std = np.std(img[mask])
    
    if std < eps:
        img[mask] = 0.0
        return img
        
    img[mask] = (img[mask] - mean) / (std + eps)
    return img

if __name__ == "__main__":
    print("Preprocess module successfully defined.")