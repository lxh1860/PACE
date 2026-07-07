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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pace_framework import combine_spectral_data, train_pace

# ---------- 1. SSDC-DA: Data Augmentation ----------
aug_patches, aug_labels, aug_sims, aug_coords = combine_spectral_data(
    image=image, image_true=image, gt=gt, gt_train=gt_train,
    patch_size=7, num_classes=num_classes,
    similar=0.85, spatial_radius=5,
)

# ---------- 2. Build DataLoaders ----------
raw_loader = DataLoader(TensorDataset(raw_images, raw_labels), batch_size=50)
aug_loader = DataLoader(TensorDataset(torch.tensor(aug_patches), torch.tensor(aug_labels)), batch_size=50)
aug_sims_loader = DataLoader(TensorDataset(torch.tensor(aug_sims).unsqueeze(1)), batch_size=50)

# ---------- 3. PGDS: Train with progressive gradient descent ----------
train_pace(
    network=model,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    train_loader_raw=raw_loader,
    val_loader=val_loader,
    epoch=350,
    saving_path="./checkpoints",
    device=device,
    smooth_loss=smooth_loss,
    train_class_num=train_class_num,
    num_class=num_classes,
    train_loader_exp=aug_loader,
    train_loader_exp_cosine_similarity=aug_sims_loader,
    num_train=50,
    begin_tip=35,
    max_grad_norm=5.0,
    scheduler=scheduler,
)
```

The `train_pace()` function implements the full PGDS training loop internally:
- **Stage 1** (warmup, `e <= begin_tip`): all augmented + raw batches
- **Stage 2** (decay, `e > begin_tip`): combined batch count decreases by 1 per epoch
- **Stage 3** (finetune, `combined == 1`): cycle through 1 raw batch per epoch

Gradient clipping (`max_grad_norm`) is applied every step (IOC-AFM).

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
