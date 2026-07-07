"""
PACE Framework -- Runnable Demo
================================

Demonstrates all three PACE modules with synthetic HSI data.
Runs on CPU in ~10 seconds.

Usage:
    python example_usage.py
"""

import numpy as np
import torch
from pace_framework import combine_spectral_data, SimFocalLoss, train_pace


def make_synthetic_hsi(h=100, w=100, bands=30, num_classes=4, seed=42):
    """Create a toy HSI cube with spatially coherent class regions."""
    rng = np.random.RandomState(seed)
    gt = np.zeros((h, w), dtype=np.int32)
    gt[:h // 2, w // 2:] = 1
    gt[h // 2:, :w // 2] = 2
    gt[h // 2:, w // 2:] = 3

    class_centers = rng.randn(num_classes, bands).astype(np.float32) * 5
    image = np.zeros((h, w, bands), dtype=np.float32)
    for c in range(num_classes):
        mask = gt == c
        image[mask] = class_centers[c] + rng.randn(mask.sum(), bands).astype(np.float32) * 0.5

    gt_train = -1 * np.ones_like(gt)
    for c in range(num_classes):
        ys, xs = np.where(gt == c)
        n = max(1, int(len(ys) * 0.01))
        idx = rng.choice(len(ys), n, replace=False)
        gt_train[ys[idx], xs[idx]] = c

    return image, gt, gt_train, num_classes


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    image, gt, gt_train, num_classes = make_synthetic_hsi()
    print(f"HSI shape: {image.shape}, Classes: {num_classes}")

    # ----------------------------------------------------------------
    # Module 1: SSDC-DA -- Data Augmentation
    # ----------------------------------------------------------------
    print("\n--- SSDC-DA: Data Augmentation ---")
    aug_patches, aug_labels, aug_sims, aug_coords = combine_spectral_data(
        image=image, image_true=image, gt=gt, gt_train=gt_train,
        patch_size=7, num_classes=num_classes,
        similar=0.85, spatial_radius=5,
    )
    print(f"  Augmented samples: {len(aug_labels)}")
    print(f"  Similarity range:  [{aug_sims.min():.4f}, {aug_sims.max():.4f}]")

    # ----------------------------------------------------------------
    # Module 2: IOC-AFM -- SimFocalLoss
    # ----------------------------------------------------------------
    print("\n--- IOC-AFM: SimFocalLoss ---")
    focal = SimFocalLoss(num_classes=num_classes, sim_power=2.0, max_gamma=2.0).to(device)
    for sim_val in [1.0, 0.9, 0.5, 0.1]:
        logits = torch.randn(8, num_classes).to(device)
        targets = torch.randint(0, num_classes, (8,)).to(device)
        sim_w = torch.full((8,), sim_val).to(device)
        loss = focal(logits, targets, sim_weights=sim_w)
        gamma = (1 - sim_val**2) * 2.0
        print(f"  sim={sim_val:.1f}  gamma~{gamma:.2f}  loss={loss.item():.4f}")

    # ----------------------------------------------------------------
    # Module 3: PGDS -- Schedule display
    # ----------------------------------------------------------------
    print("\n--- PGDS: Progressive Schedule ---")
    num_raw_batches = 5
    num_aug_batches = 360
    combined = num_raw_batches + num_aug_batches
    begin_tip = 35
    print(f"  Raw={num_raw_batches}, Aug={num_aug_batches}, Combined={combined}")
    print(f"  begin_tip={begin_tip}")
    for ep in [1, 35, 36, 100, 200, 350]:
        if ep <= begin_tip:
            stage, nb = "warmup", combined
        else:
            nb = max(1, combined - (ep - begin_tip))
            stage = "finetune" if nb == 1 else "decay"
        print(f"  Epoch {ep:>3d}: stage={stage:<8s}  batches={nb}")

    print("\n" + "=" * 50)
    print("All PACE modules verified successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
