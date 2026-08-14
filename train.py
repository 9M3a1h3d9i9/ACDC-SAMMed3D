import os
import sys
import time
from pathlib import Path

# اضافه کردن مسیر پروژه به sys.path
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from src.dataset import ACDCSAMMed3DDataset
from src.model import load_sam_med3d_model
from src.metrics import calculate_dice, calculate_iou


# ==========================================
# تعریف تابع Loss باینری اختصاصی برای SAM
# ==========================================
class BinaryDiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BinaryDiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # inputs: خروجی مدل خام (بدون سیگموید)
        # targets: لیبل‌های واقعی باینری شده (۰ و ۱)
        
        # محاسبه Binary Cross Entropy
        bce_loss = self.bce(inputs, targets)
        
        # محاسبه Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        smooth = 1e-5
        inputs_flat = inputs_sigmoid.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1.0 - dice_score
        
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
# ==========================================


def forward_sam_med3d(model, images, bboxes=None):
    """
    اجرای مدل به صورت Fully Automatic (بدون پرامپت جعبه)
    """
    image_embeddings = model.image_encoder(images)

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


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    scaler = GradScaler('cuda') if device == 'cuda' else None

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # 🌟 حیاتی‌ترین بخش: تبدیل لیبل‌های چندکلاسه (0,1,2,3) به باینری (0 و 1)
        # همچنین فرمت لیبل باید Float باشد تا با خروجی مدل همخوانی داشته باشد
        labels = (labels > 0).float()

        optimizer.zero_grad()

        if device == 'cuda':
            with autocast('cuda'):
                outputs = forward_sam_med3d(model, images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = forward_sam_med3d(model, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
            dice = calculate_dice(preds, labels)
            total_dice += dice.item() if isinstance(dice, torch.Tensor) else dice

    return total_loss / len(dataloader), total_dice / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # 🌟 تبدیل به باینری در فاز ارزیابی
            labels = (labels > 0).float()

            if device == 'cuda':
                with autocast('cuda'):
                    outputs = forward_sam_med3d(model, images)
                    loss = criterion(outputs, labels)
            else:
                outputs = forward_sam_med3d(model, images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()

            dice = calculate_dice(preds, labels)
            iou = calculate_iou(preds, labels)

            total_dice += dice.item() if isinstance(dice, torch.Tensor) else dice
            total_iou += iou.item() if isinstance(iou, torch.Tensor) else iou

    return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader)


def main():
    DATA_DIR = PROJECT_ROOT / "data" / "ACDC" / "database" / "training"
    CKPT_DIR = PROJECT_ROOT / "ckpt"
    PRETRAINED_CKPT = PROJECT_ROOT.parent / "SAM_Med3D" / "ckpt" / "sam_med3d_turbo.pth"

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 1
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    FREEZE_ENCODER = True

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Beginning SAM-Med3D training on device: {DEVICE}")

    full_dataset = ACDCSAMMed3DDataset(data_dir=str(DATA_DIR))

    if len(full_dataset) == 0:
        print("❌ Execution stopped: No training data found.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"📊 Data loaded: {len(train_dataset)} train samples | {len(val_dataset)} val samples")

    model = load_sam_med3d_model(
        checkpoint_path=str(PRETRAINED_CKPT),
        device=DEVICE,
        freeze_image_encoder=FREEZE_ENCODER
    )

    # 🌟 استفاده از تابع Loss باینری جدید که جایگزین CombinedDiceCELoss شد
    criterion = BinaryDiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_dice = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, DEVICE)

        scheduler.step()
        elapsed_time = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({elapsed_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} - Val Dice: {val_dice:.4f} - Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = CKPT_DIR / "best_sam_med3d_acdc.pth"
            torch.save(model.state_dict(), best_path)
            print(f"🔥 New best model saved! Best Val Dice: {best_val_dice:.4f} -> {best_path.name}")

    print("🎉 Training completed successfully.")


if __name__ == "__main__":
    main()