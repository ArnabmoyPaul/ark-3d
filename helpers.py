"""
helpers.py — Training utilities for Ark+ 3D.

Functions:
  cosine_lr_schedule  : cosine decay with linear warmup
  save_checkpoint     : save model + optimizer state
  load_checkpoint     : restore from checkpoint, return start_epoch
  set_seed            : global reproducibility
  get_device          : auto-detect CUDA/CPU
  estimate_vram       : print VRAM estimate for planning
"""

import os
import math
import random
import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr_schedule(
    epoch: int,
    total_epochs: int,
    lr_max: float,
    lr_min: float = None,
    warmup_epochs: int = 5,
) -> float:
    """
    Cosine decay with linear warmup.

    Warmup  : epochs [0, warmup_epochs)  → LR linearly 0 → lr_max
    Cosine  : epochs [warmup_epochs, T)  → LR cosine lr_max → lr_min

    Args:
        epoch         : current epoch (0-indexed)
        total_epochs  : total training epochs
        lr_max        : peak learning rate
        lr_min        : minimum LR (default: lr_max * 0.01)
        warmup_epochs : number of linear warmup epochs

    Returns:
        float: learning rate for this epoch
    """
    if lr_min is None:
        lr_min = lr_max * 0.01

    if epoch < warmup_epochs:
        # Linear warmup
        return lr_max * (epoch + 1) / max(warmup_epochs, 1)
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + (lr_max - lr_min) * cosine_decay


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    extra: dict = None,
) -> None:
    """
    Save full training state to disk.

    Args:
        model     : model (state_dict saved)
        optimizer : optimizer (state_dict saved)
        epoch     : current epoch
        path      : output file path (.pt or .pth)
        extra     : any additional dict to include in checkpoint
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None,
    strict: bool = True,
) -> int:
    """
    Load checkpoint and restore model (and optionally optimizer) state.

    Args:
        path      : checkpoint file path
        model     : model to restore
        optimizer : optimizer to restore (optional)
        device    : map_location device
        strict    : whether to require exact key match

    Returns:
        int: start_epoch (epoch + 1 from checkpoint, or 0 if not found)
    """
    if not os.path.exists(path):
        logger.warning(f"Checkpoint not found: {path}, starting from scratch")
        return 0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    logger.info(f"Model weights loaded from {path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state restored")

    start_epoch = checkpoint.get("epoch", -1) + 1
    logger.info(f"Resuming from epoch {start_epoch}")
    return start_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility.
    Covers Python, NumPy, PyTorch (CPU + CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic ops (slight perf hit, acceptable for research)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def get_device(verbose: bool = True) -> torch.device:
    """
    Auto-detect best available device (CUDA > CPU).

    Prints device info and VRAM if verbose=True.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"Device  : {props.name}")
            print(f"VRAM    : {vram_gb:.1f} GB")
            print(f"CUDA    : {torch.version.cuda}")
            if vram_gb < 6.0:
                print("⚠  VRAM < 6 GB: reduce batch_size to 4")
            elif vram_gb >= 8.0:
                print("✓  RTX 4060 8GB detected — batch_size=8 recommended")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Device  : CPU (no CUDA detected)")
            print("⚠  Training on CPU will be very slow — GPU strongly recommended")

    return device


# ─────────────────────────────────────────────────────────────────────────────
# VRAM estimator
# ─────────────────────────────────────────────────────────────────────────────

def estimate_vram_mb(
    batch_size: int = 8,
    depth: int = 28,
    h: int = 28,
    w: int = 28,
    embed_dim: int = 768,
    swin_model: str = "swin_tiny",
) -> None:
    """
    Print a rough VRAM budget breakdown.
    Useful for deciding batch_size before running on RTX 4060.
    """
    # After bilinear upsample to 224×224 per slice
    # Each slice: (3, 224, 224) float32 = 3*224*224*4 bytes
    slice_bytes = 3 * 224 * 224 * 4
    total_slices = batch_size * depth
    slice_mb = (total_slices * slice_bytes) / (1024 ** 2)

    # Swin-Tiny feature map ≈ 7×7×768 activations per slice
    feat_mb = (total_slices * 7 * 7 * embed_dim * 4) / (1024 ** 2)

    # Depth features: (batch_size, depth, embed_dim)
    depth_feat_mb = (batch_size * depth * embed_dim * 4) / (1024 ** 2)

    # Model params: swin_tiny ≈ 28M params, ~4×28M*4 bytes for params+grad
    model_mb = 4 * 28e6 * 4 / (1024 ** 2)

    total_estimate = slice_mb + feat_mb + depth_feat_mb + model_mb

    print(f"\nVRAM estimate (batch_size={batch_size}, depth={depth}):")
    print(f"  Input slices (upsampled 224³)  : {slice_mb:.0f} MB")
    print(f"  Swin feature maps              : {feat_mb:.0f} MB")
    print(f"  Depth aggregator features      : {depth_feat_mb:.0f} MB")
    print(f"  Model params + gradients       : {model_mb:.0f} MB")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total estimate                 : {total_estimate:.0f} MB  ({total_estimate/1024:.1f} GB)")
    print(f"  RTX 4060 budget                : 8192 MB  (8.0 GB)")
    if total_estimate < 6000:
        print(f"  ✓  Should fit comfortably at batch_size={batch_size}")
    elif total_estimate < 7500:
        print(f"  ⚠  Tight — enable AMP and watch for OOM")
    else:
        print(f"  ✗  Likely OOM — reduce batch_size to {batch_size // 2}")


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    estimate_vram_mb(batch_size=8)
    estimate_vram_mb(batch_size=4)

    # LR schedule sanity check
    lrs = [cosine_lr_schedule(e, 100, 1e-4, warmup_epochs=5) for e in range(100)]
    print(f"\nLR schedule: warmup end={lrs[4]:.2e}, peak={lrs[5]:.2e}, final={lrs[-1]:.2e}")
