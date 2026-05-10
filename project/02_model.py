"""
02_model.py
===========
Dataset klasa, augmentacije, model i loss funkcija.
EG1800 konvencija: bijela=osoba → mask > 127 = 1.0 (BEZ invertiranja)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class PortraitDataset(Dataset):
    """
    EG1800 dataset.
    Maska konvencija: bijela (255) = osoba, crna (0) = pozadina.
    """

    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.transform  = transform

        all_imgs = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Zadržaj samo parove s odgovarajućom maskom
        self.image_files = []
        for f in sorted(all_imgs):
            stem = os.path.splitext(f)[0]
            if os.path.exists(os.path.join(masks_dir, stem + ".png")):
                self.image_files.append(f)

        print(f"  PortraitDataset: {len(self.image_files)} valjanih parova")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name = self.image_files[idx]
        stem = os.path.splitext(name)[0]

        img_path  = os.path.join(self.images_dir, name)
        mask_path = os.path.join(self.masks_dir,  stem + ".png")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # EG1800: bijela=osoba → (mask > 127) = 1.0  ✓
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"].unsqueeze(0)

        return image, mask


# ─────────────────────────────────────────────
# AUGMENTACIJE
# ─────────────────────────────────────────────

def get_transforms(train: bool = True, img_size: int = 512):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.6),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15,
                               rotate_limit=20, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], is_check_shapes=False)
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], is_check_shapes=False)


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """BCE + Dice loss, jednake težine."""

    def __init__(self):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, pred, target):
        return 0.5 * self.bce(pred, target) + 0.5 * self.dice(pred, target)


# ─────────────────────────────────────────────
# METRIKE
# ─────────────────────────────────────────────

def compute_metrics(pred_logits, target, threshold: float = 0.5, eps: float = 1e-6):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    iou  = ((tp + eps) / (tp + fp + fn + eps)).mean().item()
    dice = ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean().item()
    return iou, dice


# ─────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────

def build_model(model_name: str = "deeplabv3plus",
                encoder: str = "resnet50",
                pretrained: bool = True) -> nn.Module:
    weights = "imagenet" if pretrained else None

    if model_name == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=1,
        )
    elif model_name == "unet":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=1,
            decoder_attention_type="scse",
        )
    elif model_name == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=1,
        )
    else:
        raise ValueError(f"Nepoznat model: {model_name}")

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_name} ({encoder}) | Parametri: {n:,}")
    return model
