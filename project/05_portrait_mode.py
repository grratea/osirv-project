"""
05_portrait_mode.py
===================
Portrait Mode efekt na vlastitim fotografijama.

Pokretanje:
    python 05_portrait_mode.py --image moja_slika.jpg
    python 05_portrait_mode.py --image moja_slika.jpg --blur 30
    python 05_portrait_mode.py --image moja_slika.jpg --compare   # usporedi blur radijuse
    python 05_portrait_mode.py --folder moje_slike/               # batch obrada
"""

import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

from model_utils import build_model


# ─────────────────────────────────────────────
# SEGMENTACIJA
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", "deeplabv3plus")
    encoder    = ckpt.get("encoder",    "resnet50")
    img_size   = ckpt.get("img_size",   512)

    model = build_model(model_name, encoder, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"✓ Model učitan: {model_name}/{encoder} (img_size={img_size})")
    print(f"  Dataset: {ckpt.get('dataset', '?')} | Konvencija: {ckpt.get('mask_convention', '?')}")
    return model, img_size


def segment_image(image_rgb: np.ndarray, model, device, img_size: int = 512) -> np.ndarray:
    """
    Segmentira osobu na slici.
    Vraća masku gdje 1.0 = osoba, 0.0 = pozadina.
    """
    h, w = image_rgb.shape[:2]
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)

    inp = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        raw = torch.sigmoid(model(inp))[0, 0].cpu().numpy()

    mask = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
    return mask  # EG1800 model: visoka vrijednost = osoba ✓


# ─────────────────────────────────────────────
# BOKEH BLUR
# ─────────────────────────────────────────────

def apply_bokeh(image_rgb: np.ndarray, radius: int = 20) -> np.ndarray:
    """Kružni disk blur — simulacija velikog otvora blende."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (x ** 2 + y ** 2 <= radius ** 2).astype(np.float32)
    kernel /= kernel.sum()
    blurred = np.zeros_like(image_rgb, dtype=np.float32)
    for c in range(3):
        blurred[:, :, c] = cv2.filter2D(
            image_rgb[:, :, c].astype(np.float32), -1, kernel
        )
    return blurred.astype(np.uint8)


def portrait_mode(image_rgb: np.ndarray, mask: np.ndarray,
                  blur_radius: int = 20, feather: int = 31) -> np.ndarray:
    """
    Primijeni portrait mode efekt.
    mask: 1.0=osoba (ostaje oštro), 0.0=pozadina (zamućuje se)
    feather: veličina Gaussian kernela za feathering rubova (mora biti neparno)
    """
    blurred     = apply_bokeh(image_rgb, radius=blur_radius)
    feather     = feather if feather % 2 == 1 else feather + 1
    mask_smooth = cv2.GaussianBlur(mask, (feather, feather), 0)
    alpha       = mask_smooth[:, :, np.newaxis]
    result      = alpha * image_rgb.astype(np.float32) + (1 - alpha) * blurred.astype(np.float32)
    return result.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# PRIKAZ I SPREMANJE
# ─────────────────────────────────────────────

def process_image(img_path: str, model, device, img_size: int,
                  blur_radius: int = 25, compare: bool = False,
                  output_dir: str = "output"):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"❌ Ne mogu otvoriti: {img_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    stem    = Path(img_path).stem
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Segmentiram: {img_path} ({img_rgb.shape[1]}x{img_rgb.shape[0]})")
    mask   = segment_image(img_rgb, model, device, img_size)
    result = portrait_mode(img_rgb, mask, blur_radius=blur_radius)

    # Spremi rezultat
    out_path = os.path.join(output_dir, f"{stem}_portrait.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"  ✓ Spremljeno: {out_path}")

    if compare:
        # Usporedi različite blur radijuse
        radii = [5, 10, 20, 35, 50]
        fig, axes = plt.subplots(1, len(radii) + 1, figsize=(4 * (len(radii) + 1), 5))
        fig.suptitle(f"Usporedba Bokeh intenziteta — {stem}", fontsize=13, fontweight="bold")
        axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis("off")
        for i, r in enumerate(radii):
            res = portrait_mode(img_rgb, mask, blur_radius=r)
            axes[i + 1].imshow(res)
            axes[i + 1].set_title(f"radius={r}")
            axes[i + 1].axis("off")
        plt.tight_layout()
        cmp_path = os.path.join(output_dir, f"{stem}_comparison.png")
        plt.savefig(cmp_path, dpi=120, bbox_inches="tight")
        plt.show()
        print(f"  ✓ Usporedba: {cmp_path}")
    else:
        # Standardni prikaz: original | maska | rezultat
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Portrait Mode — {stem}", fontsize=14, fontweight="bold")
        axes[0].imshow(img_rgb)
        axes[0].set_title("Originalna slika"); axes[0].axis("off")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Segmentacijska maska\n(bijela=osoba)"); axes[1].axis("off")
        axes[2].imshow(result)
        axes[2].set_title(f"Portrait Mode (blur={blur_radius})"); axes[2].axis("off")
        plt.tight_layout()
        vis_path = os.path.join(output_dir, f"{stem}_result.png")
        plt.savefig(vis_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  ✓ Vizualizacija: {vis_path}")


def process_folder(folder: str, model, device, img_size: int,
                   blur_radius: int = 25, output_dir: str = "output"):
    exts   = {".jpg", ".jpeg", ".png", ".webp"}
    images = [f for f in Path(folder).iterdir() if f.suffix.lower() in exts]
    print(f"  Pronađeno {len(images)} slika u: {folder}")

    for img_path in images:
        process_image(str(img_path), model, device, img_size,
                      blur_radius=blur_radius, output_dir=output_dir)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Portrait Mode efekt")
    p.add_argument("--image",      type=str, default=None,
                   help="Putanja do jedne slike")
    p.add_argument("--folder",     type=str, default=None,
                   help="Mapa sa slikama (batch obrada)")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    p.add_argument("--blur",       type=int, default=25,
                   help="Bokeh blur radius (5=lagano, 25=srednje, 50=jako)")
    p.add_argument("--compare",    action="store_true",
                   help="Prikaži usporedbu svih blur radijusa")
    p.add_argument("--output_dir", type=str, default="output")
    cfg = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uređaj: {device}")

    model, img_size = load_model(cfg.checkpoint, device)

    if cfg.image:
        process_image(cfg.image, model, device, img_size,
                      blur_radius=cfg.blur, compare=cfg.compare,
                      output_dir=cfg.output_dir)
    elif cfg.folder:
        process_folder(cfg.folder, model, device, img_size,
                       blur_radius=cfg.blur, output_dir=cfg.output_dir)
    else:
        print("Navedi --image ili --folder")
        print("Primjer: python 05_portrait_mode.py --image moja_slika.jpg --blur 25")
