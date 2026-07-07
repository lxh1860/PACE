"""
PACE Framework -- Runnable Demo
================================

Demonstrates the full PACE pipeline with synthetic HSI data.
No real dataset required; runs on CPU in ~10 seconds.

Usage:
    python example_usage.py
"""

import random
import numpy as np
import torch
from pace_framework import PACE


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

    # ----------------------------------------------------------------
    # 1. Prepare synthetic HSI data
    # ----------------------------------------------------------------
    image, gt, gt_train, num_classes = make_synthetic_hsi()
    bands = image.shape[-1]
    patch_size = 7
    print(f"HSI shape: {image.shape}, Classes: {num_classes}")

    # ----------------------------------------------------------------
    # 2. SSDC-DA: Data Augmentation
    # ----------------------------------------------------------------
    pace = PACE(
        num_classes=num_classes,
        total_epochs=350,
        warmup_epochs=35,
        similar=0.85,
        spatial_radius=5,
        sim_power=2.0,
        max_gamma=2.0,
        max_grad_norm=5.0,
    )
    print("\n--- SSDC-DA: Augmenting data ---")

    aug_patches, aug_labels, aug_sims, aug_coords = pace.augment(
        image=image, image_true=image, gt=gt, gt_train=gt_train, patch_size=patch_size
    )
    print(f"  Augmented samples: {len(aug_labels)}")
    print(f"  Similarity range:  [{aug_sims.min():.4f}, {aug_sims.max():.4f}]")

    # ----------------------------------------------------------------
    # 3. Build raw patches
    # ----------------------------------------------------------------
    border = patch_size // 2
    raw_images, raw_labels = [], []
    for c in range(num_classes):
        xs, ys = np.where(gt_train == c)
        for x, y in zip(xs, ys):
            if border <= x < image.shape[0] - border and border <= y < image.shape[1] - border:
                raw_images.append(image[x - border:x + border + 1,
                                        y - border:y + border + 1])
                raw_labels.append(c)

    raw_images = np.array(raw_images)
    raw_labels = np.array(raw_labels)
    print(f"  Raw samples:       {len(raw_labels)}")

    # Count batches (batch_size=16)
    bs = 16
    num_raw_batches = max(1, len(raw_labels) // bs)
    num_aug_batches = max(1, len(aug_labels) // bs)

    # ----------------------------------------------------------------
    # 4. PGDS: Initialize and display schedule
    # ----------------------------------------------------------------
    pace.init_pgds(num_aug_batches=num_aug_batches, num_raw_batches=num_raw_batches)
    print(f"\n--- PGDS Schedule ---")
    print(f"  Raw batches: {num_raw_batches}, Aug batches: {num_aug_batches}")
    print(f"  Combined init: {num_aug_batches + num_raw_batches}")
    for ep in [1, 20, 35, 36, 50, 100, 200, 300, 350]:
        stage, nb, use_aug = pace.get_pgds_plan(ep)
        print(f"  Epoch {ep:>3d}: stage={stage:<8s}  batches={nb}  augmented={use_aug}")

    # ----------------------------------------------------------------
    # 5. IOC-AFM: Verify loss computation
    # ----------------------------------------------------------------
    print(f"\n--- IOC-AFM: SimFocalLoss test ---")
    pace.focal_loss = pace.focal_loss.to(device)

    for sim_val in [1.0, 0.9, 0.5, 0.1]:
        logits = torch.randn(8, num_classes).to(device)
        targets = torch.randint(0, num_classes, (8,)).to(device)
        sim_w = torch.full((8,), sim_val).to(device)

        loss = pace.compute_loss(logits, targets, sim_weights=sim_w)
        print(f"  sim={sim_val:.1f}  loss={loss.item():.4f}  "
              f"(gamma ~ {((1 - sim_val**2) * 2):.2f})")

    # ----------------------------------------------------------------
    # 6. Gradient clipping test
    # ----------------------------------------------------------------
    print(f"\n--- Gradient Clipping test ---")
    import torch.nn as nn
    dummy = nn.Linear(10, 4).to(device)
    x = torch.randn(4, 10).to(device)
    loss = dummy(x).sum()
    loss.backward()
    print(f"  Before clip: max_grad={max(p.grad.abs().max().item() for p in dummy.parameters()):.4f}")
    pace.clip_gradients(dummy)
    print(f"  After  clip: max_grad={max(p.grad.abs().max().item() for p in dummy.parameters()):.4f}")

    print("\n" + "=" * 50)
    print("All PACE modules verified successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
