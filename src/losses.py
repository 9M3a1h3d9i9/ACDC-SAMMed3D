import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# اضافه کردن مسیر ریشه پروژه به sys.path
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class DiceLoss(nn.Module):
    """
    3D Dice Loss implementation for multi-class or binary segmentation.
    """
    def __init__(self, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, D, H, W] خروجی خام شبکه
            targets: [B, 1, D, H, W] یا [B, D, H, W] برچسب‌های واقعی
        """
        num_classes = logits.shape[1]

        if num_classes == 1:
            probs = torch.sigmoid(logits)
            targets_one_hot = targets.float()
        else:
            probs = F.softmax(logits, dim=1)
            if targets.ndim == 5 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
            targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


class CombinedDiceCELoss(nn.Module):
    """
     ترکیب تابع زیان Cross-Entropy و Dice Loss برای آموزش دقیق تصاویر ۳ بعدی پزشکی.
    """
    def __init__(self, weight_ce: float = 1.0, weight_dice: float = 1.0, smooth: float = 1e-5):
        super(CombinedDiceCELoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # تنظیم ابعاد targets برای CrossEntropyLoss
        if targets.ndim == 5 and targets.shape[1] == 1:
            ce_targets = targets.squeeze(1).long()
        else:
            ce_targets = targets.long()

        loss_ce = self.ce_loss(logits, ce_targets)
        loss_dice = self.dice_loss(logits, targets)

        total_loss = (self.weight_ce * loss_ce) + (self.weight_dice * loss_dice)
        return total_loss


if __name__ == "__main__":
    print("✅ Module src/losses.py loaded successfully.")