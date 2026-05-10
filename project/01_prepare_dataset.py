"""
01_prepare_dataset.py
=====================
Priprema EG1800 dataset za trening.

EG1800 struktura (iz ZIP-a):
    EG1800/
        GT_png/          ← maske  (bijela=osoba, crna=pozadina)
        images_data_crop/ ← slike
        EG1800_train.txt  ← lista train fajlova
        EG1800_val.txt    ← lista val fajlova
        train.txt
        val.txt

Pokretanje:
    python 01_prepare_dataset.py --eg1800_dir "C:/putanja/do/EG1800"
"""

import os
import shutil
import argparse
import cv2
import numpy as np
import glob
from pathlib import Path


def prepare_eg1800(eg1800_dir: str, output_dir: str = "dataset"):
    eg1800_dir = Path(eg1800_dir)

    # Pronađi mape
    img_dir  = eg1800_dir / "images_data_crop"
    mask_dir = eg1800_dir / "GT_png"

    if not img_dir.exists():
        # Pokušaj recursive pretragu
        found = list(eg1800_dir.rglob("images_data_crop"))
        if found:
            img_dir = found[0]
        else:
            print(f"❌ Ne mogu pronaći 'images_data_crop' u: {eg1800_dir}")
            print("   Provjeri putanju i strukturu ZIP-a")
            return 0

    if not mask_dir.exists():
        found = list(eg1800_dir.rglob("GT_png"))
        if found:
            mask_dir = found[0]
        else:
            print(f"❌ Ne mogu pronaći 'GT_png' u: {eg1800_dir}")
            return 0

    print(f"✓ Slike: {img_dir}")
    print(f"✓ Maske: {mask_dir}")

    # Pronađi sve slike
    img_files = sorted(
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.jpeg"))
    )
    print(f"  Pronađeno slika: {len(img_files)}")

    # Kreiraj output direktorije
    out_img  = Path(output_dir) / "images"
    out_mask = Path(output_dir) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    saved   = 0
    skipped = 0

    for img_path in img_files:
        stem = img_path.stem

        # Traži masku — EG1800 koristi format: 00001_mask.png
        mask_path = None
        for candidate in [
            mask_dir / (stem + "_mask.png"),   # EG1800: 00001_mask.png  ← ovo
            mask_dir / (stem + ".png"),         # standardno
            mask_dir / (stem + "_mask.jpg"),
            mask_dir / (stem + ".jpg"),
        ]:
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            skipped += 1
            continue

        # Učitaj
        img  = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            skipped += 1
            continue

        # EG1800 konvencija: bijela=osoba → (mask > 127) = 1.0  ✓
        # Provjera: ako je prosječna vrijednost > 200, maska je vjerojatno čisto bijela (greška)
        mean_val = mask.mean()
        if mean_val > 250 or mean_val < 1:
            skipped += 1
            continue

        # Spremi s originalnim imenom
        dst_img  = out_img  / (stem + ".jpg")
        dst_mask = out_mask / (stem + ".png")
        cv2.imwrite(str(dst_img),  img)
        cv2.imwrite(str(dst_mask), mask)
        saved += 1

    print(f"\n✓ Kopirano: {saved} parova")
    print(f"  Preskočeno: {skipped}")
    print(f"  Ukupno u dataset/: {len(list(out_img.glob('*')))}")

    # Provjeri txt fajlove (EG1800_train.txt, EG1800_val.txt)
    train_txt = eg1800_dir / "EG1800_train.txt"
    val_txt   = eg1800_dir / "EG1800_val.txt"
    if train_txt.exists():
        with open(train_txt) as f:
            train_names = [l.strip().split()[0] for l in f if l.strip()]
        print(f"\n  EG1800_train.txt: {len(train_names)} uzoraka")
    if val_txt.exists():
        with open(val_txt) as f:
            val_names = [l.strip().split()[0] for l in f if l.strip()]
        print(f"  EG1800_val.txt:   {len(val_names)} uzoraka")

    return saved


def verify_dataset(dataset_dir: str = "dataset", n_samples: int = 4):
    """Vizualna provjera dataseta."""
    import matplotlib.pyplot as plt
    import random

    img_files = sorted(glob.glob(f"{dataset_dir}/images/*.jpg"))
    if not img_files:
        print("❌ Nema slika u dataset/images/")
        return

    print(f"\n✓ Ukupno slika: {len(img_files)}")
    samples = random.sample(img_files, min(n_samples, len(img_files)))

    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]
    fig.suptitle("Provjera EG1800 dataseta\n(zeleno overlay = detektirana osoba ✓)",
                 fontsize=13, fontweight="bold")

    all_means = []
    for i, img_path in enumerate(samples):
        stem      = Path(img_path).stem
        mask_path = f"{dataset_dir}/masks/{stem}.png"

        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        all_means.append(mask.mean())

        # Zeleni overlay
        overlay  = img.copy().astype(np.float32)
        m_norm   = mask / 255.0
        overlay[:, :, 1] = np.clip(overlay[:, :, 1] + 120 * m_norm, 0, 255)

        axes[i][0].imshow(img);  axes[i][0].set_title(stem[:20]); axes[i][0].axis("off")
        axes[i][1].imshow(mask, cmap="gray")
        axes[i][1].set_title(f"Maska (mean={mask.mean():.0f})"); axes[i][1].axis("off")
        axes[i][2].imshow(overlay.astype(np.uint8))
        axes[i][2].set_title("Zeleno = osoba"); axes[i][2].axis("off")

    plt.tight_layout()
    plt.savefig("dataset_provjera.png", dpi=100, bbox_inches="tight")
    plt.show()

    avg = sum(all_means) / len(all_means)
    print(f"Prosječna svjetlina maski: {avg:.1f}/255")
    if 30 < avg < 180:
        print("✓ Maske izgledaju ispravno (bijela=osoba)")
    else:
        print("⚠️  Neočekivana vrijednost — provjeri vizualno")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Priprema EG1800 dataseta")
    parser.add_argument("--eg1800_dir", type=str, required=True,
                        help="Putanja do EG1800 mape (koja sadrži GT_png/ i images_data_crop/)")
    parser.add_argument("--output_dir", type=str, default="dataset",
                        help="Izlazna mapa (default: dataset/)")
    parser.add_argument("--verify", action="store_true",
                        help="Prikaži vizualnu provjeru nakon kopiranja")
    args = parser.parse_args()

    print("=" * 50)
    print("  EG1800 Dataset Priprema")
    print("=" * 50)
    n = prepare_eg1800(args.eg1800_dir, args.output_dir)
    if n > 0 and args.verify:
        verify_dataset(args.output_dir)
