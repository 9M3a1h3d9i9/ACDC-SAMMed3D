import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset import ACDCSAMMed3DDataset
from src.model import load_sam_med3d_model
from src.metrics import calculate_dice, calculate_iou


def forward_sam_med3d(model, images, bboxes=None):
    """
    اجرای مدل به صورت Fully Automatic (بدون پرامپت جعبه) 
    برای جلوگیری از باگ PromptEncoder دو بعدی در کتابخانه medim
    """
    image_embeddings = model.image_encoder(images)

    # ارسال مقدار None برای تمامی پرامپت‌ها (آموزش کاملاً خودکار)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=None, 
        masks=None,
    )

    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    if low_res_masks.shape[2:] != images.shape[2:]:
        low_res_masks = F.interpolate(
            low_res_masks,
            size=images.shape[2:],
            mode="trilinear",
            align_corners=False
        )

    return low_res_masks


def plot_slice_comparison(image_3d, gt_3d, pred_3d, save_path, slice_idx=None):
    if slice_idx is None:
        slice_idx = image_3d.shape[0] // 2

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
    # مستقیماً مسیر درست را می‌دهیم و تغییرش نمی‌دهیم
    DATA_DIR = PROJECT_ROOT / "data" / "ACDC" / "database" / "training"
    # if not any(DATA_DIR.glob("*.nii*")):
        #     DATA_DIR = PROJECT_ROOT / "data" / "raw"

    MODEL_PATH = PROJECT_ROOT / "ckpt" / "best_sam_med3d_acdc.pth"
    OUTPUT_DIR = PROJECT_ROOT / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 Running evaluation on device: {DEVICE}")

    full_dataset = ACDCSAMMed3DDataset(data_dir=str(DATA_DIR))
    if len(full_dataset) == 0:
        print("❌ No evaluation data found.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = load_sam_med3d_model(
        checkpoint_path=str(MODEL_PATH if MODEL_PATH.exists() else PROJECT_ROOT.parent / "SAM_Med3D" / "ckpt" / "sam_med3d_turbo.pth"),
        device=DEVICE
    )
    model.eval()

    total_dice = 0.0
    total_iou = 0.0

    print("📊 Calculating metrics and generating visual outputs...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            bboxes = batch["bbox"].to(DEVICE)
            filename = batch["filename"][0]

            if DEVICE == 'cuda':
                with autocast('cuda'):
                    outputs = forward_sam_med3d(model, images, bboxes)
            else:
                outputs = forward_sam_med3d(model, images, bboxes)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            dice = calculate_dice(preds, labels)
            iou = calculate_iou(preds, labels)

            total_dice += dice.item() if isinstance(dice, torch.Tensor) else dice
            total_iou += iou.item() if isinstance(iou, torch.Tensor) else iou

            preds_np = preds.squeeze().cpu().numpy()
            img_np = images.squeeze().cpu().numpy()
            gt_np = labels.squeeze().cpu().numpy()

            save_img_path = OUTPUT_DIR / f"pred_{filename.split('.')[0]}.png"
            plot_slice_comparison(img_np, gt_np, preds_np, save_img_path)

            print(f"[{i+1}/{len(test_loader)}] File: {filename} | Dice: {dice:.4f} | IoU: {iou:.4f}")

    avg_dice = total_dice / len(test_loader)
    avg_iou = total_iou / len(test_loader)

    print("\n==========================================")
    print(f"🎯 Average Dice Score: {avg_dice:.4f}")
    print(f"🎯 Average IoU Score:  {avg_iou:.4f}")
    print(f"📁 Visual outputs saved to:\n   {OUTPUT_DIR}")
    print("==========================================")


if __name__ == "__main__":
    evaluate_model()