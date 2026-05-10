"""
03_train.py
===========
Treniranje Portrait Segmentation modela na EG1800 datasetu.

Pokretanje:
    python 03_train.py
    python 03_train.py --epochs 80 --batch_size 4 --model unet
    python 03_train.py --resume  # nastavi trening od zadnjeg checkpointa
"""

import os
import json
import argparse
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _02_model import PortraitDataset, get_transforms, CombinedLoss, compute_metrics, build_model


# ─────────────────────────────────────────────
# KONFIGURACIJA
# ─────────────────────────────────────────────

def get_config():
    p = argparse.ArgumentParser(description="Trening Portrait Segmentation")
    p.add_argument("--images_dir",  default="dataset/images")
    p.add_argument("--masks_dir",   default="dataset/masks")
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--model",       default="deeplabv3plus",
                   choices=["deeplabv3plus", "unet", "unetplusplus"])
    p.add_argument("--encoder",     default="resnet50",
                   help="resnet50 | resnet34 | mobilenet_v2 | efficientnet-b3")
    p.add_argument("--img_size",    type=int, default=512)
    p.add_argument("--batch_size",  type=int, default=4,
                   help="Smanji na 2 ako imaš malo VRAM-a")
    p.add_argument("--epochs",      type=int, default=60)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--val_split",   type=float, default=0.1)
    p.add_argument("--test_split",  type=float, default=0.1)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--resume",      action="store_true",
                   help="Nastavi od zadnjeg checkpointa")
    p.add_argument("--no_cuda",     action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────
# TRAIN / VALIDATE PETLJE
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss, total_iou = 0.0, 0.0

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=80)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast("cuda"):
                out  = model(imgs)
                loss = loss_fn(out, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(imgs)
            loss = loss_fn(out, masks)
            loss.backward()
            optimizer.step()

        iou, _ = compute_metrics(out.detach(), masks)
        total_loss += loss.item()
        total_iou  += iou
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

    n = len(loader)
    return total_loss / n, total_iou / n


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0

    for imgs, masks in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
        imgs, masks = imgs.to(device), masks.to(device)
        out  = model(imgs)
        loss = loss_fn(out, masks)
        iou, dice = compute_metrics(out, masks)
        total_loss += loss.item()
        total_iou  += iou
        total_dice += dice

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


# ─────────────────────────────────────────────
# GRAFOVI
# ─────────────────────────────────────────────

def save_plots(history: dict, save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    er = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(er, history["train_loss"], label="Train")
    axes[0].plot(er, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(er, history["train_iou"], label="Train")
    axes[1].plot(er, history["val_iou"],   label="Val")
    axes[1].set_title("IoU"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(er, history["val_dice"], "g-", label="Val Dice")
    axes[2].set_title("Dice"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=120)
    plt.close()
    print(f"  Grafovi: {save_dir}/training_curves.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"
    )
    print(f"\n{'='*55}")
    print(f"  Portrait Segmentation — Trening")
    print(f"{'='*55}")
    print(f"  Uređaj:    {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Model:     {cfg.model} / {cfg.encoder}")
    print(f"  Epohe:     {cfg.epochs} | Batch: {cfg.batch_size} | LR: {cfg.lr}")
    print(f"  Img size:  {cfg.img_size}x{cfg.img_size}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────
    print("\n[1/4] Učitavam dataset...")
    full_ds = PortraitDataset(cfg.images_dir, cfg.masks_dir)
    n       = len(full_ds)
    val_n   = max(1, int(n * cfg.val_split))
    test_n  = max(1, int(n * cfg.test_split))
    train_n = n - val_n - test_n

    train_ds, val_ds, test_ds = random_split(
        full_ds, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    train_ds.dataset.transform = get_transforms(train=True,  img_size=cfg.img_size)
    val_ds.dataset.transform   = get_transforms(train=False, img_size=cfg.img_size)
    test_ds.dataset.transform  = get_transforms(train=False, img_size=cfg.img_size)

    # Windows: num_workers > 0 uzrokuje pickle greške → koristimo 0
    workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=workers, pin_memory=device.type == "cuda",
                              persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=workers, pin_memory=device.type == "cuda",
                              persistent_workers=False)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

    print(f"  Train: {train_n} | Val: {val_n} | Test: {test_n}")

    # Spremi test indekse za evaluaciju
    torch.save(test_ds.indices, os.path.join(cfg.save_dir, "test_indices.pt"))

    # ── Model ────────────────────────────────
    print("\n[2/4] Gradim model...")
    model = build_model(cfg.model, cfg.encoder).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    loss_fn   = CombinedLoss()
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 1
    best_iou    = 0.0
    history     = {k: [] for k in ["train_loss", "val_loss", "train_iou", "val_iou", "val_dice"]}

    # ── Resume ───────────────────────────────
    last_ckpt = os.path.join(cfg.save_dir, "last_checkpoint.pth")
    if cfg.resume and os.path.exists(last_ckpt):
        print(f"\n[Resume] Učitavam: {last_ckpt}")
        ckpt        = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_iou    = ckpt["best_iou"]
        history     = ckpt.get("history", history)
        print(f"  Nastavljam od epohe {start_epoch}, best IoU dosad: {best_iou:.4f}")

    # ── Trening ──────────────────────────────
    print(f"\n[3/4] Trening ({cfg.epochs} epoha)...\n")
    print(f"{'Ep':<6}{'Train L':<12}{'Val L':<12}{'Train IoU':<12}{'Val IoU':<11}{'Val Dice':<10}{'Info'}")
    print("─" * 70)

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()

        t_loss, t_iou = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        v_loss, v_iou, v_dice = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        history["train_loss"].append(t_loss); history["val_loss"].append(v_loss)
        history["train_iou"].append(t_iou);   history["val_iou"].append(v_iou)
        history["val_dice"].append(v_dice)

        info = ""
        if v_iou > best_iou:
            best_iou = v_iou
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_iou":      v_iou,
                "val_dice":     v_dice,
                "model_name":   cfg.model,
                "encoder":      cfg.encoder,
                "img_size":     cfg.img_size,
                "dataset":      "EG1800",
                "mask_convention": "white=person (mask > 127)",
            }, os.path.join(cfg.save_dir, "best_model.pth"))
            info = "✓ best"

        # Spremi zadnji checkpoint (za resume)
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_iou":        best_iou,
            "history":         history,
        }, last_ckpt)

        elapsed = time.time() - t0
        print(f"{epoch:<6}{t_loss:<12.4f}{v_loss:<12.4f}{t_iou:<12.4f}{v_iou:<11.4f}{v_dice:<10.4f}  {info}  ({elapsed:.0f}s)")

        # Spremi grafove svakih 10 epoha
        if epoch % 10 == 0:
            save_plots(history, cfg.save_dir)

    print(f"\n{'='*55}")
    print(f"  ✓ Trening završen! Best Val IoU: {best_iou:.4f}")
    print(f"  Model: {cfg.save_dir}/best_model.pth")
    print(f"{'='*55}")

    save_plots(history, cfg.save_dir)

    # Spremi history kao JSON
    with open(os.path.join(cfg.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
