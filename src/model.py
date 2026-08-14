import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

# ۱. اضافه کردن مسیر ریشه پروژه و مسیر SAM_Med3D به sys.path
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# بررسی و اضافه کردن پوشه مجاور SAM_Med3D در صورت وجود
SAM_MED3D_PATH = PROJECT_ROOT.parent / "SAM_Med3D"
if SAM_MED3D_PATH.exists() and str(SAM_MED3D_PATH) not in sys.path:
    sys.path.append(str(SAM_MED3D_PATH))


def load_sam_med3d_model(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    freeze_image_encoder: bool = False
):
    """
    مدل SAM-Med3D را بارگذاری کرده و وزن‌های sam_med3d_turbo.pth را روی آن قرار می‌دهد.

    Args:
        checkpoint_path (str): مسیر کامل فایل sam_med3d_turbo.pth
        device (str): دستگاه اجرا (cuda یا cpu)
        freeze_image_encoder (bool): آیا وزن‌های Image Encoder انجماد (Freeze) شوند یا خیر.

    Returns:
        nn.Module: مدل آماده برای Fine-tuning
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        # raise FileNotFoundError(f"❌ فایل وزن‌ها در مسیر زیر پیدا نشد:\n{ckpt_path}")
        raise FileNotFoundError(f"❌ Weights file not found at:\n{ckpt_path}")

    # print(f"📦 در حال بارگذاری مدل SAM-Med3D از مسیر:\n{ckpt_path}")
    print(f"📦 Loading SAM-Med3D model from:\n{ckpt_path}")

    # روش اول: استفاده از کتابخانه medim (در صورت نصب بودن)
    try:
        import medim
        model = medim.create_model("SAM-Med3D", pretrained=False)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        # print("✅ مدل با موفقیت از طریق پکیج 'medim' بارگذاری شد.")
        print(f"📦 Loading SAM-Med3D model from:\n{ckpt_path}")

    except Exception as e1:
        # روش دوم: بارگذاری مستقیم از سورس‌کد SAM_Med3D
        try:
            from segment_anything.build_sam3D import sam_model_registry3D
            model = sam_model_registry3D["vit_b_ori"](checkpoint=str(ckpt_path))
            # print("✅ مدل با موفقیت از طریق سورس‌کد 'segment_anything' بارگذاری شد.")
            print(f"📦 Loading successfully SAM-Med3D from:\n{ckpt_path}")

        except Exception as e2:
            # raise RuntimeError(
            #     f"❌ خطا در بارگذاری مدل! مطمئن شوید پکیج medim نصب است یا پوشه SAM_Med3D کنار پروژه وجود دارد.\n"
            #     f"خطای اول: {e1}\nخطای دوم: {e2}"
            # )
            raise RuntimeError(
                f"❌ Error loading model! Ensure 'medim' package is installed or 'SAM_Med3D' folder is present next to the project.\n"
                f"First error: {e1}\nSecond error: {e2}"
            )

    # مدیریت انجماد وزن‌ها (Freeze vs Full Fine-tuning)
    if freeze_image_encoder:
        # print("🔒 بخش Image Encoder منجمد شد (فقط Prompt Encoder و Mask Decoder آموزش می‌بینند).")
        print("🔒 Image Encoder is frozen (Only Prompt Encoder and Mask Decoder will be trained).")
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    else:
        # print("🔓 تمام پارامترهای مدل قابل آموزش هستند (حالت Full Fine-tuning).")
        print("🔓 All model parameters are trainable (Full Fine-tuning mode).")

    model = model.to(device)
    return model


if __name__ == "__main__":
    # تست سریع ماژول
    print("✅ Module src/model.py loaded successfully.")