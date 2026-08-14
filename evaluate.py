import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset import ACDCSAMMed3DDataset
from src.model import load_sam_med3d_model
from src.metrics import calculate_dice, calculate_iou


def plot_slice_comparison(image_3d, gt_3d, pred_3d, save_path, slice_idx=None):
    """ذخیره تصویر مقایسه‌ای از یک اسلایس از اسکن ۳ بعدی"""
    if slice_idx is None:
        slice_idx = image_3d.shape[0] // 2  # انتخاب اسلایس میانی

    img_slice = image_3d[slice_idx]
    gt_slice = gt_3d[slice_idx]
    pred_slice = pred_3d[slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title("Input MRI Slice")
    axes[0].axis("off")

    axes[1].imshow(gt_slice, cmap="jet")
    axes[1].set_title("Ground Truth (GT)")
    axes[1].axis("off")

    axes[2].imshow(pred_slice, cmap="jet")
    axes[2].set_title("SAM-Med3D Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_model():
    DATA_DIR = PROJECT_ROOT / "data" / "ACDC" / "database" / "training"
    if not any(DATA_DIR.glob("*.nii*")):
        DATA_DIR = PROJECT_ROOT / "data" / "raw"

    MODEL_PATH = PROJECT_ROOT / "ckpt" / "best_sam_med3d_acdc.pth"
    OUTPUT_DIR = PROJECT_ROOT / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"🔍 اجرای ارزیابی روی دستگاه: {DEVICE}")
    print(f"🔍 Running evaluation on device: {DEVICE}")

    # test_dataset = ACDCSAMMed3DDataset(data_dir=str(DATA_DIR))
    # if len(test_dataset) == 0:
    #     print("❌ هیچ داده‌ای برای ارزیابی یافت نشد.")
        # print("❌ No evaluation data found.")
    #     return

    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # ------------------------------------------------------------------------------------
    # برای جلوگیری از ارزیابی روی همه داده‌ها، میتوانید از همان random_split استفاده کنید
    full_dataset = ACDCSAMMed3DDataset(data_dir=str(DATA_DIR))
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42) # استفاده از سید مشترک برای انتخاب همان داده‌های قبلی
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # بارگذاری مدل
    model = load_sam_med3d_model(
        checkpoint_path=str(MODEL_PATH if MODEL_PATH.exists() else PROJECT_ROOT.parent / "SAM_Med3D" / "ckpt" / "sam_med3d_turbo.pth"),
        device=DEVICE
    )
    model.eval()

    total_dice = 0.0
    total_iou = 0.0

    # print("📊 در حال محاسبه متریژها و تولید تصاویر خروجی...")
    print("📊 Calculating metrics and generating visual outputs...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            bboxes = batch["bbox"].to(DEVICE)
            filename = batch["filename"][0]

            try:
                outputs = model(images, bboxes)
            except TypeError:
                outputs = model(images)

            dice = calculate_dice(outputs, labels)
            iou = calculate_iou(outputs, labels)

            total_dice += dice
            total_iou += iou

            # تبدیل خروجی‌ها به حالت ۳ بعدی برای رسم
            if outputs.ndim == 5 and outputs.shape[1] > 1:
                preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            else:
                preds = (torch.sigmoid(outputs) > 0.5).squeeze().cpu().numpy()

            img_np = images.squeeze().cpu().numpy()
            gt_np = labels.squeeze().cpu().numpy()

            # ذخیره خروجی تصویری نمونه
            save_img_path = OUTPUT_DIR / f"pred_{filename.split('.')[0]}.png"
            plot_slice_comparison(img_np, gt_np, preds, save_img_path)

            print(f"[{i+1}/{len(test_loader)}] File: {filename} | Dice: {dice:.4f} | IoU: {iou:.4f}")

    avg_dice = total_dice / len(test_loader)
    avg_iou = total_iou / len(test_loader)

    print("\n==========================================")
    # print(f"🎯 میانگین شاخص Dice: {avg_dice:.4f}")
    print(f"🎯 Average Dice Score: {avg_dice:.4f}")
    # print(f"🎯 میانگین شاخص IoU:  {avg_iou:.4f}")
    print(f"🎯 Average IoU Score:  {avg_iou:.4f}")
    # print(f"📁 خروجی‌های تصویری در پوشه ذخیره شدند:\n   {OUTPUT_DIR}")
    print(f"📁 Visual outputs saved to:\n   {OUTPUT_DIR}")
    print("==========================================")


if __name__ == "__main__":
    evaluate_model()