"""
metrics.py — Evaluation metrics and logging for Ark+ 3D.

Functions:
  compute_auroc   : per-class AUROC, handles single-class edge case
  compute_accuracy: top-1 accuracy
  log_metrics     : append metrics dict to CSV
  plot_training_curves : matplotlib training/val curves
"""

import os
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AUROC
# ─────────────────────────────────────────────────────────────────────────────

def compute_auroc(targets: np.ndarray, probs: np.ndarray) -> float:
    """
    Compute AUROC for binary classification.

    Args:
        targets : (N,) int/float — ground truth binary labels {0, 1}
        probs   : (N,) float     — predicted probabilities for positive class

    Returns:
        float AUC in [0, 1]. Returns 0.5 on edge cases (single class in targets).
    """
    try:
        unique_classes = np.unique(targets)
        if len(unique_classes) < 2:
            logger.warning(
                f"compute_auroc: only one class present ({unique_classes}), "
                "returning 0.5 (chance level)"
            )
            return 0.5
        return float(roc_auc_score(targets.astype(int), probs))
    except Exception as e:
        logger.warning(f"compute_auroc failed: {e}, returning 0.5")
        return 0.5


def compute_multiclass_auroc(targets: np.ndarray, probs: np.ndarray) -> float:
    """
    Macro-averaged OvR AUROC for multi-class.

    Args:
        targets : (N,) int     — ground truth class indices
        probs   : (N, C) float — softmax probabilities

    Returns:
        float macro-AUROC
    """
    try:
        n_classes = probs.shape[1]
        unique_classes = np.unique(targets)
        if len(unique_classes) < 2:
            return 0.5

        aucs = []
        for c in range(n_classes):
            if c not in unique_classes:
                continue
            binary_targets = (targets == c).astype(int)
            if binary_targets.sum() == 0 or (1 - binary_targets).sum() == 0:
                continue
            auc = roc_auc_score(binary_targets, probs[:, c])
            aucs.append(auc)

        return float(np.mean(aucs)) if aucs else 0.5
    except Exception as e:
        logger.warning(f"compute_multiclass_auroc failed: {e}, returning 0.5")
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(targets: np.ndarray, preds: np.ndarray) -> float:
    """
    Top-1 accuracy.

    Args:
        targets : (N,) int — ground truth class indices
        preds   : (N,) int — predicted class indices

    Returns:
        float accuracy in [0, 1]
    """
    if len(targets) == 0:
        return 0.0
    return float(np.mean(targets.astype(int) == preds.astype(int)))


# ─────────────────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────────────────

def log_metrics(epoch: int, metrics: dict, save_path: str) -> None:
    """
    Append a metrics dict to a CSV file. Creates header row if file is new.

    Args:
        epoch     : current epoch (int, 0-indexed)
        metrics   : dict of metric_name → float
        save_path : path to CSV file
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fieldnames = list(metrics.keys())
    write_header = not os.path.exists(save_path)

    with open(save_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                         for k, v in metrics.items()})


# ─────────────────────────────────────────────────────────────────────────────
# Training curve visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str = None) -> None:
    """
    Plot training and validation curves from history dict.

    Args:
        history   : dict returned by train_ark_plus()
        save_path : if given, save figure to this path; else show interactively
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Ark+ 3D Training Curves", fontsize=14, fontweight="bold")

    # ── Train loss ─────────────────────────────────────────────────────────
    ax = axes[0, 0]
    if "train_loss" in history:
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], "b-", linewidth=1.5, label="Train Loss")
    if "cons_loss" in history:
        ax.plot(epochs, history["cons_loss"], "r--", linewidth=1.0, label="Consistency Loss")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Learning rate ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    if "lr" in history:
        epochs = range(1, len(history["lr"]) + 1)
        ax.plot(epochs, history["lr"], "g-", linewidth=1.5)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # ── EMA momentum ──────────────────────────────────────────────────────
    ax = axes[1, 0]
    if "momentum" in history:
        epochs = range(1, len(history["momentum"]) + 1)
        ax.plot(epochs, history["momentum"], "m-", linewidth=1.5)
    ax.set_title("Teacher EMA Momentum")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Momentum")
    ax.set_ylim([0.99, 1.001])
    ax.grid(True, alpha=0.3)

    # ── Validation metrics ─────────────────────────────────────────────────
    ax = axes[1, 1]
    val_keys = [k for k in history if "_val_" in k]
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(val_keys), 1)))
    for i, key in enumerate(val_keys):
        vals = history[key]
        ep = range(1, len(vals) + 1)
        short_name = key.replace("ds", "DS").replace("_val_", " ")
        ax.plot(ep, vals, "-o", color=colors[i], linewidth=1.5,
                markersize=4, label=short_name)
    ax.set_title("Validation Metrics")
    ax.set_xlabel("Eval epoch")
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training curves saved to {save_path}")
    else:
        plt.show()

    return fig


def print_final_summary(test_metrics: dict, datasets_names: list,
                        n_classes_list: list) -> None:
    """Print a clean summary table of final test results."""
    print("\n" + "=" * 55)
    print(f"{'Dataset':<22} {'Metric':<8} {'Score':>8}")
    print("-" * 55)
    for i, name in enumerate(datasets_names):
        if n_classes_list[i] == 2:
            key = f"ds{i}_test_auc"
            metric = "AUC"
        else:
            key = f"ds{i}_test_acc"
            metric = "ACC"
        score = test_metrics.get(key, 0.0)
        print(f"{name:<22} {metric:<8} {score:>8.4f}")
    print("=" * 55)
