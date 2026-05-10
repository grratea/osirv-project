# """
# 04_evaluate.py
# ==============
# Evaluacija modela na testnom skupu + vizualizacija predikcija.

# Pokretanje:
#     python 04_evaluate.py
#     python 04_evaluate.py --checkpoint checkpoints/best_model.pth
# """

# import os
# import argparse
# import numpy as np
# import torch
# import cv2
# import matplotlib.pyplot as plt
# import segmentation_models_pytorch as smp
# from torch.utils.data import DataLoader, Subset

# from model_utils import PortraitDataset, get_transforms, build_model


# def evaluate(cfg):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\n{'='*50}")
#     print(f"  Evaluacija modela")
#     print(f"{'='*50}")
#     print(f"  Checkpoint: {cfg.checkpoint}")
#     print(f"  Uređaj: {device}")

#     # ── Učitaj model ──────────────────────────────
#     ckpt = torch.load(cfg.checkpoint, map_location=device)
#     model_name = ckpt.get("model_name", "deeplabv3plus")
#     encoder    = ckpt.get("encoder",    "resnet50")
#     img_size   = ckpt.get("img_size",   512)

#     print(f"\n  Model: {model_name} / {encoder} | Img size: {img_size}")
#     print(f"  Treniran u epohi: {ckpt.get('epoch', '?')}")
#     print(f"  Val IoU (val skup): {ckpt.get('val_iou', '?'):.4f}")
#     print(f"  Konvencija maske: {ckpt.get('mask_convention', 'nepoznata')}")

#     model = build_model(model_name, encoder, pretrained=False)
#     model.load_state_dict(ckpt["model_state"])
#     model = model.to(device).eval()

#     # ── Dataset ───────────────────────────────────
#     full_ds = PortraitDataset(cfg.images_dir, cfg.masks_dir,
#                                transform=get_transforms(train=False, img_size=img_size))

#     # Pokušaj učitati test indekse iz treninga
#     test_idx_path = os.path.join(os.path.dirname(cfg.checkpoint), "test_indices.pt")
#     if os.path.exists(test_idx_path):
#         test_indices = torch.load(test_idx_path)
#         test_ds = Subset(full_ds, test_indices)
#         print(f"  Test skup (iz treninga): {len(test_ds)} uzoraka")
#     else:
#         # Fallback: uzmi zadnjih 10%
#         n = len(full_ds)
#         test_ds = Subset(full_ds, list(range(int(n * 0.9), n)))
#         print(f"  Test skup (fallback 10%): {len(test_ds)} uzoraka")

#     loader = DataLoader(test_ds, batch_size=1, shuffle=False)

#     # ── Metrike ───────────────────────────────────
#     all_iou, all_dice, all_pa = [], [], []

#     with torch.no_grad():
#         for imgs, masks in loader:
#             imgs, masks = imgs.to(device), masks.to(device)
#             out  = torch.sigmoid(model(imgs))
#             pred = (out > 0.5).float()

#             for i in range(imgs.shape[0]):
#                 p = pred[i, 0].cpu().numpy()
#                 t = masks[i, 0].cpu().numpy()
#                 inter = (p * t).sum()
#                 union = p.sum() + t.sum() - inter
#                 all_iou.append((inter + 1e-6) / (union + 1e-6))
#                 all_dice.append((2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6))
#                 all_pa.append((p == t).mean())

#     print(f"\n{'='*50}")
#     print(f"  REZULTATI NA TESTNOM SKUPU ({len(all_iou)} uzoraka)")
#     print(f"{'='*50}")
#     print(f"  IoU  (mean ± std):  {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
#     print(f"  Dice (mean):        {np.mean(all_dice):.4f}")
#     print(f"  Pixel Accuracy:     {np.mean(all_pa):.4f}")
#     print(f"{'='*50}")

#     # ── Vizualizacija ─────────────────────────────
#     samples = []
#     with torch.no_grad():
#         for imgs, masks in loader:
#             imgs, masks = imgs.to(device), masks.to(device)
#             out = torch.sigmoid(model(imgs))
#             for i in range(imgs.shape[0]):
#                 img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
#                 img_np = (img_np * np.array([0.229, 0.224, 0.225]) +
#                           np.array([0.485, 0.456, 0.406])).clip(0, 1)
#                 samples.append((
#                     img_np,
#                     masks[i, 0].cpu().numpy(),
#                     out[i, 0].cpu().numpy()
#                 ))
#                 if len(samples) >= cfg.n_samples:
#                     break
#             if len(samples) >= cfg.n_samples:
#                 break

#     n = len(samples)
#     fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
#     if n == 1:
#         axes = [axes]
#     fig.suptitle(
#         f"Test predikcije | IoU={np.mean(all_iou):.4f} | Dice={np.mean(all_dice):.4f}",
#         fontsize=13, fontweight="bold"
#     )

#     for i, (img, gt, pred) in enumerate(samples):
#         iou_i = (pred > 0.5).astype(float)
#         inter = (iou_i * gt).sum()
#         union = iou_i.sum() + gt.sum() - inter
#         iou_val = (inter + 1e-6) / (union + 1e-6)

#         axes[i][0].imshow(img);           axes[i][0].set_title("Originalna");      axes[i][0].axis("off")
#         axes[i][1].imshow(gt, cmap="gray"); axes[i][1].set_title("Ground Truth");   axes[i][1].axis("off")
#         axes[i][2].imshow(pred, cmap="gray", vmin=0, vmax=1)
#         axes[i][2].set_title(f"Predikcija (IoU={iou_val:.3f})"); axes[i][2].axis("off")

#     plt.tight_layout()
#     out_path = "test_predictions.png"
#     plt.savefig(out_path, dpi=120, bbox_inches="tight")
#     plt.show()
#     print(f"\n  Vizualizacija: {out_path}")


# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--checkpoint",  default="checkpoints/best_model.pth")
#     p.add_argument("--images_dir",  default="dataset/images")
#     p.add_argument("--masks_dir",   default="dataset/masks")
#     p.add_argument("--n_samples",   type=int, default=6)
#     cfg = p.parse_args()
#     evaluate(cfg)


"""
04_evaluate.py
==============
Evaluacija modela na testnom skupu + vizualizacija predikcija.

Pokretanje:
    python 04_evaluate.py
    python 04_evaluate.py --checkpoint checkpoints/best_model.pth
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from _02_model import PortraitDataset, get_transforms, build_model


def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Evaluacija modela")
    print(f"{'='*50}")
    print(f"  Checkpoint: {cfg.checkpoint}")
    print(f"  Uređaj: {device}")

    # ── Učitaj model ──────────────────────────────
    ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "deeplabv3plus")
    encoder    = ckpt.get("encoder",    "resnet50")
    img_size   = ckpt.get("img_size",   512)

    print(f"\n  Model: {model_name} / {encoder} | Img size: {img_size}")
    print(f"  Treniran u epohi: {ckpt.get('epoch', '?')}")
    print(f"  Val IoU (val skup): {ckpt.get('val_iou', '?'):.4f}")
    print(f"  Konvencija maske: {ckpt.get('mask_convention', 'nepoznata')}")

    model = build_model(model_name, encoder, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    # ── Dataset ───────────────────────────────────
    full_ds = PortraitDataset(cfg.images_dir, cfg.masks_dir,
                               transform=get_transforms(train=False, img_size=img_size))

    test_idx_path = os.path.join(os.path.dirname(cfg.checkpoint), "test_indices.pt")
    if os.path.exists(test_idx_path):
        test_indices = torch.load(test_idx_path, weights_only=False)
        test_ds = Subset(full_ds, test_indices)
        print(f"  Test skup (iz treninga): {len(test_ds)} uzoraka")
    else:
        n = len(full_ds)
        test_ds = Subset(full_ds, list(range(int(n * 0.9), n)))
        print(f"  Test skup (fallback 10%): {len(test_ds)} uzoraka")

    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # ── Metrike ───────────────────────────────────
    all_iou, all_dice, all_pa = [], [], []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out  = torch.sigmoid(model(imgs))
            pred = (out > 0.5).float()

            for i in range(imgs.shape[0]):
                p = pred[i, 0].cpu().numpy()
                t = masks[i, 0].cpu().numpy()
                inter = (p * t).sum()
                union = p.sum() + t.sum() - inter
                all_iou.append((inter + 1e-6) / (union + 1e-6))
                all_dice.append((2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6))
                all_pa.append((p == t).mean())

    print(f"\n{'='*50}")
    print(f"  REZULTATI NA TESTNOM SKUPU ({len(all_iou)} uzoraka)")
    print(f"{'='*50}")
    print(f"  IoU  (mean ± std):  {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
    print(f"  Dice (mean):        {np.mean(all_dice):.4f}")
    print(f"  Pixel Accuracy:     {np.mean(all_pa):.4f}")
    print(f"{'='*50}")

    # ── Prikupi SVE uzorke s IoU vrijednošću ──────
    all_samples = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = torch.sigmoid(model(imgs))
            for i in range(imgs.shape[0]):
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) +
                          np.array([0.485, 0.456, 0.406])).clip(0, 1)
                p_np  = out[i, 0].cpu().numpy()
                gt_np = masks[i, 0].cpu().numpy()
                inter   = ((p_np > 0.5) * gt_np).sum()
                union   = (p_np > 0.5).sum() + gt_np.sum() - inter
                iou_val = (inter + 1e-6) / (union + 1e-6)
                all_samples.append((iou_val, img_np, gt_np, p_np))

    # Sortiraj po IoU
    all_samples.sort(key=lambda x: x[0])
    worst = all_samples[:6]   # 6 najlošijih
    best  = all_samples[-6:]  # 6 najboljih

    def plot_samples(samples, title, filename):
        fig, axes = plt.subplots(6, 3, figsize=(12, 24))
        fig.suptitle(title, fontsize=13, fontweight="bold")
        for i, (iou_val, img, gt, pred) in enumerate(samples):
            iou_i = (pred > 0.5).astype(float)
            inter = (iou_i * gt).sum()
            union = iou_i.sum() + gt.sum() - inter
            iou_val = (inter + 1e-6) / (union + 1e-6)

            axes[i][0].imshow(img)
            axes[i][0].set_title(f"Originalna [{i+1}]"); axes[i][0].axis("off")
            axes[i][1].imshow(gt, cmap="gray")
            axes[i][1].set_title("Ground Truth"); axes[i][1].axis("off")
            axes[i][2].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[i][2].set_title(f"Predikcija (IoU={iou_val:.3f})"); axes[i][2].axis("off")

        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        plt.show()
        print(f"  Spremljeno: {filename}")

    # ── Vizualizacija 1: 6 najlošijih ─────────────
    plot_samples(
        worst,
        f"Najlošiji primjeri segmentacije  |  IoU (mean): {np.mean(all_iou):.4f}",
        "worst_predictions.png"
    )

    # ── Vizualizacija 2: 6 najboljih ──────────────
    plot_samples(
        best,
        f"Najbolji primjeri segmentacije  |  IoU (mean): {np.mean(all_iou):.4f}",
        "best_predictions.png"
    )

    # ── Vizualizacija 3: 6 nasumičnih ─────────────
    plot_samples(
        random.sample(all_samples, min(6, len(all_samples))),
        f"Nasumični primjeri  |  IoU={np.mean(all_iou):.4f}  |  Dice={np.mean(all_dice):.4f}",
        "test_predictions.png"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/best_model.pth")
    p.add_argument("--images_dir",  default="dataset/images")
    p.add_argument("--masks_dir",   default="dataset/masks")
    p.add_argument("--n_samples",   type=int, default=6)
    cfg = p.parse_args()
    evaluate(cfg)
