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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Module 1: SSDC-DA -- Spatial-Spectral Dual-Constraint Data Augmentation
# =============================================================================

def combine_spectral_data(image, image_true, gt, gt_train, patch_size,
                          num_classes, similar=0.85, spatial_radius=5):
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
        Higher = stricter spectral constraint = fewer but purer samples.
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

    def __init__(self, num_classes=10, sim_power=2.0, max_gamma=2.0):
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

class PACE:
    """Unified interface for the PACE training framework.

    Encapsulates SSDC-DA (data augmentation), PGDS (progressive sampling),
    and IOC-AFM (adaptive focal loss + gradient clipping).

    Parameters
    ----------
    num_classes : int
        Number of land-cover classes.
    total_epochs : int
        Total number of training epochs.
    warmup_epochs : int
        Epochs for Stage 1 (full augmented + real data).
    similar : float
        Spectral similarity threshold factor for SSDC-DA.
    spatial_radius : int
        Neighborhood radius for SSDC-DA spatial constraint.
    sim_power : float
        lambda exponent for SimFocalLoss confidence.
    max_gamma : float
        gamma_max upper bound for SimFocalLoss.
    max_grad_norm : float
        L2 gradient clipping threshold C.
    """

    def __init__(self, num_classes, total_epochs=350, warmup_epochs=35,
                 similar=0.85, spatial_radius=5,
                 sim_power=2.0, max_gamma=2.0, max_grad_norm=5.0):
        self.num_classes = num_classes
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.similar = similar
        self.spatial_radius = spatial_radius
        self.max_grad_norm = max_grad_norm

        self.focal_loss = SimFocalLoss(
            num_classes=num_classes,
            sim_power=sim_power,
            max_gamma=max_gamma,
        )

        self._num_aug_batches = 0
        self._num_raw_batches = 0
        self._min_batches = 1
        self._combined_init = 0

    # ----- SSDC-DA -----
    def augment(self, image, image_true, gt, gt_train, patch_size):
        """Run SSDC-DA to generate pseudo-labeled samples.

        Returns
        -------
        aug_patches, aug_labels, aug_sims, aug_coords
        """
        return combine_spectral_data(
            image, image_true, gt, gt_train, patch_size,
            self.num_classes, self.similar, self.spatial_radius,
        )

    # ----- PGDS -----
    def init_pgds(self, num_aug_batches, num_raw_batches):
        """Record batch counts for PGDS scheduling.

        Call once after building DataLoaders from augmented/raw data.

        Parameters
        ----------
        num_aug_batches : int
            Number of batches from the augmented dataset.
        num_raw_batches : int
            Number of batches from the original labeled dataset.
        """
        self._num_aug_batches = num_aug_batches
        self._num_raw_batches = num_raw_batches
        self._combined_init = num_aug_batches + num_raw_batches

    def get_pgds_plan(self, epoch):
        """Return the PGDS sampling plan for a given epoch.

        Three-stage schedule (faithful to original train() logic):
            warmup  (epoch <= warmup_epochs):   all augmented + raw
            decay   (epoch > warmup, size > 1):  random sample from combined
            finetune (size == 1):                cycle through 1 raw batch

        Returns
        -------
        stage : str
            One of 'warmup', 'decay', 'finetune'.
        num_batches : int
            How many batches to use this epoch.
        use_augmented : bool
            Whether augmented data participates.
        """
        # Mirror original: current_batch_size_combined -= 1 after warmup
        current = self._combined_init - max(0, epoch - self.warmup_epochs)
        current = max(self._min_batches, current)

        if epoch <= self.warmup_epochs:
            return 'warmup', self._combined_init, True
        elif current == self._min_batches:
            return 'finetune', self._min_batches, False
        else:
            return 'decay', current, True

    # ----- IOC-AFM -----
    def compute_loss(self, logits, targets, sim_weights=None):
        """Compute Sim-Adaptive Focal Loss."""
        return self.focal_loss(logits, targets, sim_weights)

    def clip_gradients(self, model):
        """Apply adaptive gradient clipping (IOC-AFM safety valve)."""
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.max_grad_norm
        )
