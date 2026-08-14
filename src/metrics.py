import sys
from pathlib import Path
import torch

# اضافه کردن مسیر ریشه پروژه به sys.path
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def calculate_dice(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = None, smooth: float = 1e-5) -> float:
    """
    محاسبه میانگین شاخص Dice Coefficient برای ارزیابی قطعه‌بندی.
    """
    if preds.ndim == 5 and preds.shape[1] > 1:
        preds = torch.argmax(preds, dim=1)
    elif preds.ndim == 5 and preds.shape[1] == 1:
        preds = (torch.sigmoid(preds) > 0.5).long()

    if targets.ndim == 5:
        targets = targets.squeeze(1)

    preds = preds.long()
    targets = targets.long()

    if num_classes is None:
        num_classes = int(max(preds.max().item(), targets.max().item()) + 1)

    dice_scores = []
    # محاسبه برای کلاس‌های پیش‌زمینه (حذف پس‌زمینه index=0)
    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = torch.sum(pred_cls * target_cls)
        cardinality = torch.sum(pred_cls) + torch.sum(target_cls)

        dice = (2.0 * intersection + smooth) / (cardinality + smooth)
        dice_scores.append(dice.item())

    return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0


def calculate_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = None, smooth: float = 1e-5) -> float:
    """
    محاسبه شاخص IoU (Intersection over Union / Jaccard Index).
    """
    if preds.ndim == 5 and preds.shape[1] > 1:
        preds = torch.argmax(preds, dim=1)
    elif preds.ndim == 5 and preds.shape[1] == 1:
        preds = (torch.sigmoid(preds) > 0.5).long()

    if targets.ndim == 5:
        targets = targets.squeeze(1)

    preds = preds.long()
    targets = targets.long()

    if num_classes is None:
        num_classes = int(max(preds.max().item(), targets.max().item()) + 1)

    iou_scores = []
    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = torch.sum(pred_cls * target_cls)
        union = torch.sum(pred_cls) + torch.sum(target_cls) - intersection

        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())

    return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0


if __name__ == "__main__":
    print("✅ Module src/metrics.py loaded successfully.")