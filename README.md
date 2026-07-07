# PACE

**P**rogressive **A**daptive **C**onstrained **E**fficient Framework

A plug-and-play data-centric training framework for efficient hyperspectral image (HSI) classification. PACE optimizes training dynamics without altering backbone architectures, achieving simultaneous improvements in both accuracy and training speed.

> **Paper**: *A Plug-and-Play Data-Centric Training Framework for Efficient Hyperspectral Image Classification* (IEEE JSTARS, under review)

## Highlights

- **Plug-and-play**: Works with any HSI classification backbone (CNN, Transformer, etc.) without structural modifications.
- **Dual advantage**: Simultaneously improves OA (+1~5%) and reduces total training time (-20~36%).
- **Physically grounded**: Leverages spatial homogeneity and spectral continuity as domain priors.

## Framework Overview

PACE consists of three synergistic modules:

| Module | Full Name | Function |
|--------|-----------|----------|
| **SSDC-DA** | Spatial-Spectral Dual-Constraint Data Augmentation | Generates physically consistent pseudo-labeled samples via spatial proximity + spectral similarity |
| **PGDS** | Progressive Gradient Descent Strategy | Three-stage curriculum learning that progressively decays augmented data, eliminating computational redundancy |
| **IOC-AFM** | Input-Output Coupled Adaptive Focusing Mechanism | Sim-Adaptive Focal Loss that maps spectral confidence to dynamic focal parameters + gradient clipping |

## Installation

```bash
git clone https://github.com/<your-username>/PACE.git
cd PACE
pip install -r requirements.txt
```

## Quick Start

```python
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pace_framework import PACE

# ---------- 1. SSDC-DA: Data Augmentation ----------
pace = PACE(
    num_classes=16, total_epochs=350, warmup_epochs=35,
    similar=0.85, spatial_radius=5,
    sim_power=2.0, max_gamma=2.0, max_grad_norm=5.0,
)

# image: (H, W, B) HSI cube; gt_train: (H, W) labels, -1 = unlabeled
aug_patches, aug_labels, aug_sims, aug_coords = pace.augment(
    image=image, image_true=image,
    gt=gt, gt_train=gt_train, patch_size=7
)

# ---------- 2. Build DataLoaders ----------
raw_loader = DataLoader(TensorDataset(raw_images, raw_labels), batch_size=50)
aug_loader = DataLoader(TensorDataset(torch.tensor(aug_patches), torch.tensor(aug_labels)), batch_size=50)
aug_sims_loader = DataLoader(TensorDataset(torch.tensor(aug_sims).unsqueeze(1)), batch_size=50)

# Convert to batch lists (same as original train() logic)
raw_list = [(imgs, lbls) for imgs, lbls in raw_loader]
raw_sims = [[1.0] * len(imgs) for imgs, _ in raw_loader]
aug_list = [(imgs, lbls) for imgs, lbls in aug_loader]
aug_sims = [s[0].tolist() for s in aug_sims_loader]

combined_list = raw_list + aug_list
combined_sims = raw_sims + aug_sims

# ---------- 3. PGDS: Initialize & Train ----------
pace.init_pgds(num_aug_batches=len(aug_list), num_raw_batches=len(raw_list))
up_tip = 0

for epoch in range(1, 351):
    stage, num_batches, use_aug = pace.get_pgds_plan(epoch)

    if stage == 'warmup':
        train_loader = combined_list
        train_sims = combined_sims
    elif stage == 'decay':
        zipped = list(zip(combined_list, combined_sims))
        selected = random.sample(zipped, num_batches)
        train_loader, train_sims = zip(*selected)
    else:  # finetune
        start, end = up_tip, up_tip + 1
        if end >= len(raw_list):
            up_tip = 0
            start = 0
            end = 1
        train_loader = raw_list[start:end]
        train_sims = raw_sims[start:end]
        up_tip = end

    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device).float(), targets.to(device).long()
        sim_w = torch.tensor(train_sims[batch_idx]).float().to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = pace.compute_loss(logits, targets, sim_weights=sim_w)
        loss.backward()
        pace.clip_gradients(model)
        optimizer.step()
```

## SimFocalLoss vs Standard Cross-Entropy

The `PACE.compute_loss()` uses **Sim-Adaptive Focal Loss** by default, which adaptively down-weights noisy pseudo-labeled samples via spectral confidence. If you prefer standard Cross-Entropy (e.g., for ablation studies), simply bypass the focal loss:

```python
# Standard CE (no IOC-AFM)
loss = torch.nn.CrossEntropyLoss()(logits, targets)

# Sim-Adaptive Focal Loss (full IOC-AFM)
loss = pace.compute_loss(logits, targets, sim_weights=sim_w)
```

Gradient clipping (`pace.clip_gradients(model)`) is recommended in both cases.

## Project Structure

```
PACE/
├── pace_framework.py   # Core framework (SSDC-DA + PGDS + IOC-AFM)
├── example_usage.py    # Runnable demo with synthetic data
├── requirements.txt
├── LICENSE
└── README.md
```

## Results

PACE was evaluated on 4 benchmark HSI datasets across 11 backbone architectures:

| Dataset | Avg OA Improvement | Avg Time Reduction |
|---------|-------------------:|-------------------:|
| SA      | +1.33%             | -35.92%            |
| PU      | +1.08%             | -21.39%            |
| WHU-LK  | +1.91%             | -22.14%            |
| HR-L    | +0.92%             | -21.13%            |

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2026pace,
  title   = {A Plug-and-Play Data-Centric Training Framework for Efficient
             Hyperspectral Image Classification},
  author  = {Li, Xinhao and Xie, Hong and Yan, Li and Xu, Mingming and
             Shen, Wenfei and Shen, Hengtong and Song, Jiang and
             Guo, Weili and Wei, Yaxuan},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations
             and Remote Sensing},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

This work was partially supported by the National Key R&D Program of China (Grant No. 2025YFB3910302), the National Natural Science Foundation of China (Grant No. 42394061), and the Zhejiang Province "Vanguard" and "Geese Leading" Research and Development Plan (Grant No. 2025C01073).
