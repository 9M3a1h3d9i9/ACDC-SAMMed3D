import os
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.cuda.amp import autocast, GradScaler # برای بهینه‌سازی حافظه GPU

# ۱. اضافه کردن مسیر ریشه پروژه به sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset import ACDCSAMMed3DDataset
from src.model import load_sam_med3d_model
from src.losses import CombinedDiceCELoss
from src.metrics import calculate_dice, calculate_iou


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    scaler = torch.amp.GradScaler('cuda')  # استفاده از نگارش جدید

    for batch in dataloader:
        images = batch["image"].to(device)  # [B, 1, D, H, W]
        labels = batch["label"].to(device)  # [B, 1, D, H, W]
        bboxes = batch["bbox"].to(device)   # [B, 6]

        optimizer.zero_grad()

        # ساخت ورودی به فرمت مورد انتظار SAM-Med3D
        batched_input = []
        for i in range(images.shape[0]):
            item = {"image": images[i]}  # حتماً کلید "image" وجود داشته باشد
            if bboxes is not None:
                item["boxes"] = bboxes[i]  # اگر نیاز به bounding box دارید
            batched_input.append(item)

        with torch.amp.autocast('cuda'):
            outputs = model(batched_input)  # فقط یک آرگومان
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
            dice = (2.0 * (preds * labels).sum()) / (preds.sum() + labels.sum() + 1e-8)
            total_dice += dice.item()

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
            bboxes = batch["bbox"].to(device)

            # ساخت batched_input به همان شکل
            batched_input = []
            for i in range(images.shape[0]):
                item = {"image": images[i]}
                if bboxes is not None:
                    item["boxes"] = bboxes[i]
                batched_input.append(item)

            outputs = model(batched_input)
            loss = criterion(outputs, labels)
            dice = calculate_dice(outputs, labels)
            iou = calculate_iou(outputs, labels)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou

    return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader)



from torch.utils.data import random_split # این خط را به بالای فایل train.py اضافه کنید

def main():
    # تنظیم مسیر دقیق پوشه training در دیتاست ACDC
    DATA_DIR = PROJECT_ROOT / "data" / "ACDC" / "database" / "training"
    CKPT_DIR = PROJECT_ROOT / "ckpt"
    PRETRAINED_CKPT = PROJECT_ROOT.parent / "SAM_Med3D" / "ckpt" / "sam_med3d_turbo.pth"
    
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # BATCH_SIZE = 2
    BATCH_SIZE = 1  # کاهش اندازه بچ برای صرفه‌جویی در حافظه VRAM
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    # FREEZE_ENCODER = False
    FREEZE_ENCODER = True  #  برای کاهش مصرف VRAM و جلوگیری از overfitting
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # print(f"🚀 شروع فرآیند آموزش SAM-Med3D روی دستگاه: {DEVICE}")
    print(f"🚀 Beggining training SAM-Med3D on device: {DEVICE}")

    # پاکسازی حافظه کش GPU پیش از شروع
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # --- ۱. بارگذاری داده‌ها ---
    full_dataset = ACDCSAMMed3DDataset(data_dir=str(DATA_DIR))

    if len(full_dataset) == 0:
        # print("❌ توقف اجرا: هیچ داده‌ای یافت نشد.")
        print(f" Stop execution: No data found. Please check the dataset path: {DATA_DIR}")
        # print(f"مسیر جستجو شده: {DATA_DIR}")
        return

    # تقسیم‌بندی ۸۰٪ به ۲۰٪ برای آموزش و اعتبارسنجی
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42) # برای تکرارپذیری نتایج
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # print(f"📊 داده‌ها بارگذاری شدند: {len(train_dataset)} نمونه آموزش | {len(val_dataset)} نمونه اعتبارسنجی")
    print(f" Datas has been loaded: {len(train_dataset)} training samples | {len(val_dataset)} validation samples")

    # --- ۲. بارگذاری مدل ---
    model = load_sam_med3d_model(
        checkpoint_path=str(PRETRAINED_CKPT),
        device=DEVICE,
        freeze_image_encoder=FREEZE_ENCODER
    )

    # --- ۳. توابع و بهینه‌ساز ---
    criterion = CombinedDiceCELoss(weight_ce=1.0, weight_dice=1.0)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # --- ۴. حلقه آموزش ---
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
            print(f" New model saved! Best Val Dice: {best_val_dice:.4f} -> {best_path.name}")

    print(" Training Successfully Completed! ✅")


if __name__ == "__main__":
    main()