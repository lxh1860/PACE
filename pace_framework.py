"""
PACE: Progressive Adaptive Constrained Efficient Framework
===========================================================

A plug-and-play data-centric training framework for efficient
hyperspectral image (HSI) classification.

Paper: "A Plug-and-Play Data-Centric Training Framework for Efficient
        Hyperspectral Image Classification"

Core Modules:
    1. SSDC-DA  -- Spatial-Spectral Dual-Constraint Data Augmentation
    2. PGDS     -- Progressive Gradient Descent Strategy
    3. IOC-AFM  -- Input-Output Coupled Adaptive Focusing Mechanism

Dependencies: numpy, torch
"""

import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
# Module 1: SSDC-DA -- Spatial-Spectral Dual-Constraint Data Augmentation
# =============================================================================

def combine_spectral_data(image, image_true, gt, gt_train, patch_size,
                          num_classes, similar, spatial_radius):
    """Generate physically consistent pseudo-labeled samples via dual constraints.

    Stage 1 (Spatial): For each labeled pixel, collect unlabeled neighbors
        within ``spatial_radius``.
    Stage 2 (Spectral): Compute per-class mean spectrum, filter candidates
        whose cosine similarity to the mean exceeds
        ``mean(intra_class_sims) * similar``.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, B)
        HSI cube used for patch extraction.
    image_true : np.ndarray, shape (H, W, B)
        HSI cube used for spectral similarity computation.
    gt : np.ndarray, shape (H, W)
        Full ground-truth map.
    gt_train : np.ndarray, shape (H, W)
        Training label map (-1 for unlabeled).
    patch_size : int
        Spatial patch size (must be odd).
    num_classes : int
        Number of land-cover classes.
    similar : float
        Spectral similarity threshold factor in (0, 1].
    spatial_radius : int
        Neighborhood radius R for spatial constraint.

    Returns
    -------
    augmented_patches : np.ndarray, shape (N, H_patch, W_patch, B)
    augmented_labels  : np.ndarray, shape (N,)
    augmented_sims    : np.ndarray, shape (N,)
    augmented_coords  : np.ndarray, shape (N, 2)
    """
    combined_images = []
    combined_labels = []
    combined_cosine_similarity = []
    expanded_coords_list = []

    border = patch_size // 2
    img_h, img_w, _ = image.shape

    for class_idx in range(num_classes):
        x_train, y_train = np.where(gt_train == class_idx)
        if len(x_train) == 0:
            continue

        train_coords_set = set(zip(x_train, y_train))

        # Representative spectrum (class mean)
        class_spectra = image_true[x_train, y_train, :]
        mean_spectrum = np.mean(class_spectra, axis=0)

        # Intra-class cosine similarity -> adaptive threshold
        std_norm = np.linalg.norm(mean_spectrum)
        class_norms = np.linalg.norm(class_spectra, axis=1)
        sims = np.zeros(len(class_spectra))
        if std_norm > 0:
            valid = class_norms > 0
            sims[valid] = (np.dot(class_spectra[valid], mean_spectrum)
                           / (class_norms[valid] * std_norm))
        threshold = np.mean(sims) * similar

        # Spatial neighborhood candidates
        candidate_coords = set()
        offsets = [(dx, dy)
                   for dx in range(-spatial_radius, spatial_radius + 1)
                   for dy in range(-spatial_radius, spatial_radius + 1)
                   if not (dx == 0 and dy == 0)]

        for cx, cy in zip(x_train, y_train):
            for dx, dy in offsets:
                nx, ny = cx + dx, cy + dy
                if border <= nx < img_h - border and border <= ny < img_w - border:
                    candidate_coords.add((nx, ny))

        candidate_coords -= train_coords_set
        if not candidate_coords:
            continue

        candidate_coords_list = list(candidate_coords)
        cand_x, cand_y = zip(*candidate_coords_list)
        cand_x = np.array(cand_x)
        cand_y = np.array(cand_y)

        # Vectorized cosine similarity
        cand_spectra = image_true[cand_x, cand_y, :]
        cand_norms = np.linalg.norm(cand_spectra, axis=1)
        cand_sims = np.zeros(len(cand_spectra))
        if std_norm > 0:
            valid = cand_norms > 0
            cand_sims[valid] = (np.dot(cand_spectra[valid], mean_spectrum)
                                / (cand_norms[valid] * std_norm))

        # Filter by spectral threshold
        valid_idx = np.where(cand_sims > threshold)[0]
        if len(valid_idx) == 0:
            continue

        final_x = cand_x[valid_idx]
        final_y = cand_y[valid_idx]
        final_sims = cand_sims[valid_idx]

        # Extract patches
        patches = []
        for fx, fy in zip(final_x, final_y):
            patch = image[fx - border:fx + border + 1,
                          fy - border:fy + border + 1, :]
            patches.append(patch)

        combined_images.append(np.array(patches))
        combined_labels.append(np.full(len(final_x), class_idx))
        combined_cosine_similarity.append(final_sims)
        expanded_coords_list.append(np.array(list(zip(final_x, final_y))))

    if not combined_images:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (np.concatenate(combined_images),
            np.concatenate(combined_labels),
            np.concatenate(combined_cosine_similarity),
            np.concatenate(expanded_coords_list))


# =============================================================================
# Module 2: IOC-AFM -- Input-Output Coupled Adaptive Focusing Mechanism
# =============================================================================

class SimFocalLoss(nn.Module):
    """Sim-Adaptive Focal Loss (core of IOC-AFM).

    Establishes a non-linear mapping from input spectral consistency
    to output focal parameters:

        confidence  alpha_i = (Sim_i) ^ sim_power
        gamma_i             = (1 - alpha_i) * max_gamma
        loss                = -(1 - p_t)^{gamma_i} * log(p_t)

    When alpha_i -> 1 (high-confidence real sample):
        gamma_i -> 0, loss degenerates to standard Cross-Entropy.
    When alpha_i -> 0 (low-confidence pseudo-labeled sample):
        gamma_i -> max_gamma, focal mechanism suppresses noisy gradients.

    Parameters
    ----------
    num_classes : int
        Number of classification classes.
    sim_power : float
        Exponent lambda for spectral confidence: alpha = Sim^lambda.
    max_gamma : float
        Upper bound gamma_max for the dynamic focusing parameter.
    """

    def __init__(self, num_classes, sim_power, max_gamma):
        super().__init__()
        self.num_classes = num_classes
        self.sim_power = sim_power
        self.max_gamma = max_gamma
        self.epsilon = 1e-7

    def forward(self, logits, targets, sim_weights=None):
        """
        Parameters
        ----------
        logits : Tensor, shape (N, C)
            Raw model outputs before softmax.
        targets : Tensor, shape (N,)
            Ground-truth class indices.
        sim_weights : Tensor or None, shape (N,)
            Per-sample spectral similarity in [0, 1].
            Real samples should use sim_weights = 1.0.
            If None, falls back to standard CE (gamma = 0).
        """
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=self.epsilon, max=1.0 - self.epsilon)
        log_probs = torch.log(probs)

        p_t = probs.gather(1, targets.view(-1, 1)).squeeze()
        log_p_t = log_probs.gather(1, targets.view(-1, 1)).squeeze()

        if sim_weights is not None:
            if sim_weights.device != logits.device:
                sim_weights = sim_weights.to(logits.device)
            confidence = sim_weights ** self.sim_power
            current_gamma = (1.0 - confidence) * self.max_gamma
        else:
            current_gamma = torch.zeros_like(p_t)

        focal_weight = (1 - p_t).pow(current_gamma).detach()
        loss = -focal_weight * log_p_t
        return loss.mean()


# =============================================================================
# Module 3: PGDS -- Progressive Gradient Descent Strategy
# =============================================================================

def random_select_batches(population, select_num):
    """Randomly sample ``select_num`` items from ``population``."""
    actual_select = max(1, min(select_num, len(population)))
    return random.sample(population, actual_select)


def train_pace(network, optimizer, criterion,
               train_loader_raw, val_loader, epoch, saving_path, device,
               smooth_loss, train_class_num, num_class,
               train_loader_exp, train_loader_exp_cosine_similarity,
               num_train, begin_tip, max_grad_norm,
               scheduler=None):
    """PGDS: Progressive Gradient Descent Strategy training loop.

    Three-stage progressive schedule:
        Stage 1 (warmup,  e <= begin_tip):    all augmented + raw batches
        Stage 2 (decay,   e > begin_tip):     combined batch count - 1 / epoch
        Stage 3 (finetune, combined == 1):    cycle through 1 raw batch / epoch

    Parameters
    ----------
    network : nn.Module
        Classification backbone.
    optimizer : torch.optim.Optimizer
    criterion : nn.Module
        Standard loss (e.g. CrossEntropyLoss).
    train_loader_raw : DataLoader
        Original labeled training data.
    val_loader : DataLoader
        Validation data.
    epoch : int
        Total training epochs.
    saving_path : str
        Checkpoint directory.
    device : torch.device
    smooth_loss : nn.Module
        Auxiliary smooth loss (used for proposed models).
    train_class_num : list
        Per-class sample counts (for CB_loss if needed).
    num_class : int
        Number of classes.
    train_loader_exp : DataLoader
        Augmented (expanded) training data from SSDC-DA.
    train_loader_exp_cosine_similarity : DataLoader
        Spectral similarity values for augmented data.
    num_train : int
        Max number of augmented batches to use.
    begin_tip : int
        Epoch at which PGDS decay begins.
    max_grad_norm : float
        Gradient clipping threshold C for IOC-AFM.
    scheduler : optional
        Learning rate scheduler.
    """
    best_acc = -0.1
    epoch_losses = []

    # --- Data preparation: convert DataLoaders to batch lists ---
    train_loader_raw_list = []
    train_loader_raw_cosine_similarity_list = []
    for i, (images, targets) in enumerate(train_loader_raw):
        train_loader_raw_list.append((images, targets))
        train_loader_raw_cosine_similarity_list.append([1.0] * len(images))

    train_loader_exp_list = []
    train_loader_exp_cosine_similarity_list = []
    for i, image_cos in enumerate(train_loader_exp_cosine_similarity):
        if i < num_train:
            train_loader_exp_cosine_similarity_list.append(image_cos)
        else:
            break

    for i, (images_exp, targets_exp) in enumerate(train_loader_exp):
        if i < num_train:
            train_loader_exp_list.append((images_exp, targets_exp))
        else:
            break

    combined_list = train_loader_raw_list + train_loader_exp_list
    combined_list_cosine_similarity = train_loader_raw_cosine_similarity_list + train_loader_exp_cosine_similarity_list

    current_batch_size_raw = len(train_loader_raw_list)
    current_batch_size_combined = len(combined_list)
    up_tip = 0
    max_curr = 1

    # --- Initialize IOC-AFM: SimFocalLoss ---
    criterion_focal = SimFocalLoss(
        num_classes=num_class,
        sim_power=2,
        max_gamma=2,
    ).to(device)

    for e in tqdm(range(1, epoch + 1), desc="training"):
        network.train()
        batch_losses = []

        # --- PGDS sampling logic ---
        start = up_tip
        end = up_tip + max_curr

        if e > begin_tip:
            current_batch_size_combined = int(current_batch_size_combined - 1)

        current_batch_size = max(max_curr, current_batch_size_combined)

        if current_batch_size == max_curr:
            # Stage 3 (finetune): cycle through 1 raw batch
            if end >= current_batch_size_raw:
                up_tip = 0
                train_loader = train_loader_raw_list[start:]
                train_loader_cos = train_loader_raw_cosine_similarity_list[start:]
            else:
                train_loader = train_loader_raw_list[start:end]
                train_loader_cos = train_loader_raw_cosine_similarity_list[start:end]
            up_tip = up_tip + max_curr
        else:
            if e <= begin_tip:
                # Stage 1 (warmup): use all data
                train_loader = combined_list
                train_loader_cos = combined_list_cosine_similarity
            else:
                # Stage 2 (decay): random sample from combined
                zipped_data = list(zip(combined_list, combined_list_cosine_similarity))
                selected_zipped = random_select_batches(zipped_data, current_batch_size)
                train_loader, train_loader_cos = zip(*selected_zipped)

        # --- Training loop ---
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device).float(), targets.to(device).long()

            optimizer.zero_grad()
            outputs = network(images)

            sim_w = torch.tensor(train_loader_cos[batch_idx]).float().to(device)

            # Loss
            loss0 = criterion(outputs, targets)

            if num_class > 0 and smooth_loss is not None:
                loss2 = smooth_loss(outputs, targets)
                loss = loss0 + 0.5 * loss2
            else:
                loss = loss0

            loss.backward()

            # IOC-AFM: Gradient clipping
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            batch_losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        if batch_losses:
            epoch_losses.append(np.mean(batch_losses))

        if e % 20 == 0 or e == 1:
            val_acc = _validation(network, val_loader, device, num_class)

        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)
        _save_checkpoint(network, is_best, saving_path, epoch=e, acc=best_acc)

    return best_acc


# =============================================================================
# Helpers
# =============================================================================

def _validation(network, val_loader, device, num_class):
    """Validation accuracy."""
    num_correct = 0.
    total_num = 0.
    network.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs, dim=1)
            num_correct += (predicted == targets).sum().item()
            total_num += len(targets)
    return num_correct / total_num if total_num > 0 else 0.0


def _save_checkpoint(network, is_best, saving_path, **kwargs):
    """Save best model checkpoint."""
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)
    if is_best:
        tqdm.write("epoch = {epoch}: best validation OA = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(saving_path, 'model_best.pth'))
