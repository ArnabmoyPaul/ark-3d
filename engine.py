"""
engine.py — Cyclic multi-task training loop for Ark+ 3D.

Core loop (Ark+ §Methods):
  For each epoch:
    Round-robin over datasets:
      Student forward  → task_loss + (optionally) consistency_loss
      Teacher forward  → consistency target (no_grad)
      EMA update after every optimizer.step()

Loss weights: 1 / ln(n_classes) to normalize gradient scale across tasks.
  binary (2):    1 / ln(2)  ≈ 1.443
  3-class:       1 / ln(3)  ≈ 0.910
  11-class:      1 / ln(11) ≈ 0.418

Consistency loss coefficient (Ark+ Eq.):
  coff = min(0.5, (m - 0.9) * 5)
  → starts near 0 when m ≈ 0.996, reaches 0.5 when m ≈ 1.0
"""

import os
import math
import time
import logging
from collections import defaultdict
from typing import List, Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ark_plus_3d.models.ark_plus import ArkPlus3D
from ark_plus_3d.utils.metrics import compute_auroc, compute_accuracy, log_metrics
from ark_plus_3d.utils.helpers import cosine_lr_schedule, save_checkpoint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Loss weight computation
# ─────────────────────────────────────────────────────────────────────────────

def task_loss_weight(n_classes: int) -> float:
    """1 / ln(n_classes) — equalises gradient scale across task sizes."""
    return 1.0 / math.log(max(n_classes, 2))


def consistency_coeff(momentum: float) -> float:
    """Ark+ formula: coff = min(0.5, (m - 0.9) * 5)."""
    return min(0.5, (momentum - 0.9) * 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch training step
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ArkPlus3D,
    train_loaders,
    n_classes_list: List[int],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    momentum_teacher: float,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    model.teacher_encoder.eval()  # teacher always in eval (BN inference)
    model.teacher_projector.eval()

    # Iterators for each dataset
    iterators = [iter(loader) for loader in train_loaders]
    max_batches = max(len(loader) for loader in train_loaders)

    # Per-dataset loss criteria
    criteria = []
    for nc in n_classes_list:
        if nc == 2:
            criteria.append(nn.BCEWithLogitsLoss())
        else:
            criteria.append(nn.CrossEntropyLoss())

    loss_weights = [task_loss_weight(nc) for nc in n_classes_list]
    mse_loss = nn.MSELoss()

    # Current EMA momentum (cosine annealed)
    m = ArkPlus3D.cosine_momentum(epoch, total_epochs, m_base=momentum_teacher)

    total_loss_sum = 0.0
    task_loss_sums = defaultdict(float)
    cons_loss_sum = 0.0
    step_count = 0

    pbar = tqdm(range(max_batches), desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)

    for _ in pbar:
        for ds_idx in range(len(train_loaders)):
            # Get next batch (cycle iterators if exhausted)
            try:
                s_view, t_view, targets = next(iterators[ds_idx])
            except StopIteration:
                iterators[ds_idx] = iter(train_loaders[ds_idx])
                s_view, t_view, targets = next(iterators[ds_idx])

            s_view  = s_view.to(device, non_blocking=True)
            t_view  = t_view.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                # ── Student forward ─────────────────────────────────────
                s_proj, s_logits = model(s_view, head_idx=ds_idx, mode="student")

                # ── Task loss ───────────────────────────────────────────
                criterion = criteria[ds_idx]
                lw = loss_weights[ds_idx]

                if n_classes_list[ds_idx] == 2:
                    # Binary: targets shape (B, 1), logits shape (B, 1) or (B, 2)
                    if s_logits.shape[-1] == 2:
                        # Use only positive class logit for BCEWithLogitsLoss
                        task_loss = criterion(s_logits[:, 1:2], targets) * lw
                    else:
                        task_loss = criterion(s_logits, targets) * lw
                else:
                    # Multi-class: targets are long scalars
                    task_loss = criterion(s_logits, targets) * lw

                # ── Consistency loss (after warmup) ─────────────────────
                if epoch >= warmup_epochs:
                    with torch.no_grad():
                        t_proj, _ = model(t_view, head_idx=ds_idx, mode="teacher")

                    coff = consistency_coeff(m)
                    cons_loss = mse_loss(s_proj, t_proj.detach())
                    total_loss = (1.0 - coff) * task_loss + coff * cons_loss
                    cons_loss_sum += cons_loss.item()
                else:
                    total_loss = task_loss

            # ── Backward ────────────────────────────────────────────────
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # ── EMA update (AFTER optimizer.step) ───────────────────────
            model.update_teacher(m)

            # ── Logging ─────────────────────────────────────────────────
            total_loss_sum += total_loss.item()
            task_loss_sums[ds_idx] += task_loss.item()
            step_count += 1

        pbar.set_postfix({"loss": f"{total_loss_sum / max(step_count, 1):.4f}",
                          "m": f"{m:.5f}"})

    n_ds = len(train_loaders)
    metrics = {
        "train_loss": total_loss_sum / max(step_count, 1),
        "cons_loss": cons_loss_sum / max(step_count, 1),
        "momentum": m,
    }
    for i in range(n_ds):
        metrics[f"task_loss_{i}"] = task_loss_sums[i] / max(step_count // n_ds, 1)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Validation / Test
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: ArkPlus3D,
    loaders,
    n_classes_list: List[int],
    device: torch.device,
    split: str = "val",
) -> Dict[str, float]:
    model.eval()
    metrics = {}

    for ds_idx, loader in enumerate(loaders):
        all_targets, all_probs = [], []
        criteria = (nn.BCEWithLogitsLoss() if n_classes_list[ds_idx] == 2
                    else nn.CrossEntropyLoss())
        total_loss = 0.0

        for s_view, _, targets in loader:
            s_view  = s_view.to(device)
            targets = targets.to(device)

            with autocast():
                _, logits = model(s_view, head_idx=ds_idx, mode="student")

            # Probabilities
            if n_classes_list[ds_idx] == 2:
                prob = torch.sigmoid(logits[:, 1:2] if logits.shape[-1] == 2
                                     else logits).squeeze(-1)
                all_probs.append(prob.cpu().float())
                all_targets.append(targets.squeeze(-1).cpu().float())
            else:
                prob = torch.softmax(logits, dim=-1)
                all_probs.append(prob.cpu().float())
                all_targets.append(targets.cpu().long())

        all_targets = torch.cat(all_targets)
        all_probs   = torch.cat(all_probs)

        if n_classes_list[ds_idx] == 2:
            auc = compute_auroc(all_targets.numpy(), all_probs.numpy())
            metrics[f"ds{ds_idx}_{split}_auc"] = auc
        else:
            preds = all_probs.argmax(dim=-1)
            acc = compute_accuracy(all_targets.numpy(), preds.numpy())
            metrics[f"ds{ds_idx}_{split}_acc"] = acc

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SOTA benchmarks
# ─────────────────────────────────────────────────────────────────────────────

SOTA_BENCHMARKS = {
    "organmnist3d":    {"metric": "ACC", "value": 0.997},
    "nodulemnist3d":   {"metric": "AUC", "value": 0.863},
    "adrenalmnist3d":  {"metric": "AUC", "value": 0.874},
    "fracturemnist3d": {"metric": "ACC", "value": 0.714},
    "vesselmnist3d":   {"metric": "AUC", "value": 0.914},
    "synapsemnist3d":  {"metric": "AUC", "value": 0.843},
}


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_ark_plus(
    model: ArkPlus3D,
    datasets_names: List[str],
    datasets_config: Dict,
    train_loaders,
    val_loaders,
    test_loaders,
    epochs: int = 100,
    lr: float = 1e-4,
    warmup_epochs: int = 5,
    momentum_teacher: float = 0.996,
    device: torch.device = None,
    save_dir: str = "./checkpoints",
    exp_name: str = "ark_3d",
    eval_every: int = 5,
) -> Dict:
    """
    Full Ark+ training loop.

    Args:
        model            : ArkPlus3D instance
        datasets_names   : list of dataset display names (keys in datasets_config)
        datasets_config  : loaded YAML config dict
        train_loaders    : list of DataLoaders for training
        val_loaders      : list of DataLoaders for validation
        test_loaders     : list of DataLoaders for test
        epochs           : total training epochs
        lr               : peak learning rate
        warmup_epochs    : linear warmup before consistency loss activates
        momentum_teacher : EMA momentum base (cosine annealed → 1.0)
        device           : torch device
        save_dir         : directory to save checkpoints and logs
        exp_name         : experiment identifier
        eval_every       : validation frequency (epochs)

    Returns:
        history dict with all per-epoch metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"{exp_name}_metrics.csv")

    # n_classes per dataset
    n_classes_list = [
        len(datasets_config[name]["diseases"]) for name in datasets_names
    ]
    # For binary tasks, force n_classes=2 for consistent loss weight
    n_classes_list = [max(nc, 2) for nc in n_classes_list]

    logger.info(f"Datasets   : {datasets_names}")
    logger.info(f"n_classes  : {n_classes_list}")
    logger.info(f"Device     : {device}")
    logger.info(f"Epochs     : {epochs}  |  LR: {lr}  |  Warmup: {warmup_epochs}")
    logger.info(f"Params     : {model.param_count()}")

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=lr,
        weight_decay=0.05,
        betas=(0.9, 0.999),
    )
    scaler = GradScaler()

    history = defaultdict(list)
    best_val_score = -1.0
    lr_min = lr * 0.01

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(epochs):
        t0 = time.time()

        # LR schedule (cosine with warmup)
        current_lr = cosine_lr_schedule(
            epoch=epoch,
            total_epochs=epochs,
            lr_max=lr,
            lr_min=lr_min,
            warmup_epochs=warmup_epochs,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loaders=train_loaders,
            n_classes_list=n_classes_list,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            total_epochs=epochs,
            warmup_epochs=warmup_epochs,
            momentum_teacher=momentum_teacher,
            device=device,
        )

        for k, v in train_metrics.items():
            history[k].append(v)
        history["lr"].append(current_lr)

        # Validate
        val_metrics = {}
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            val_metrics = evaluate(model, val_loaders, n_classes_list, device, split="val")
            for k, v in val_metrics.items():
                history[k].append(v)

            val_scores = [v for k, v in val_metrics.items() if "_val_" in k]
            mean_val = sum(val_scores) / max(len(val_scores), 1)

            if mean_val > best_val_score:
                best_val_score = mean_val
                save_checkpoint(model, optimizer, epoch,
                                os.path.join(save_dir, f"{exp_name}_best.pt"))

            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"LR={current_lr:.2e} | "
                f"Loss={train_metrics['train_loss']:.4f} | "
                f"m={train_metrics['momentum']:.5f} | "
                f"Val={mean_val:.4f} | "
                f"t={time.time()-t0:.1f}s"
            )
        else:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"LR={current_lr:.2e} | "
                f"Loss={train_metrics['train_loss']:.4f} | "
                f"m={train_metrics['momentum']:.5f} | "
                f"t={time.time()-t0:.1f}s"
            )

        # Log to CSV
        all_metrics = {"epoch": epoch + 1, **train_metrics, **val_metrics}
        log_metrics(epoch, all_metrics, log_path)

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("="*60)

    # Load best checkpoint for test eval
    best_ckpt = os.path.join(save_dir, f"{exp_name}_best.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint from epoch {ckpt.get('epoch', '?')}")

    test_metrics = evaluate(model, test_loaders, n_classes_list, device, split="test")

    for k, v in test_metrics.items():
        history[k].append(v)

    # Compare with SOTA
    flags = [datasets_config[name]["flag"] for name in datasets_names]
    logger.info("\n{:<20} {:>10} {:>10} {:>10}".format(
        "Dataset", "Metric", "Ours", "SOTA"))
    logger.info("-" * 55)

    for i, (name, flag) in enumerate(zip(datasets_names, flags)):
        metric_key = f"ds{i}_test_auc" if n_classes_list[i] == 2 else f"ds{i}_test_acc"
        our_score = test_metrics.get(metric_key, 0.0)
        sota = SOTA_BENCHMARKS.get(flag, {})
        sota_val = sota.get("value", "-")
        metric_name = sota.get("metric", "?")
        delta = f"({our_score - sota_val:+.3f})" if isinstance(sota_val, float) else ""
        logger.info(f"{name:<20} {metric_name:>10} {our_score:>10.4f} {str(sota_val):>10} {delta}")

    logger.info("="*60)

    # Save last checkpoint
    save_checkpoint(model, optimizer, epochs - 1,
                    os.path.join(save_dir, f"{exp_name}_last.pt"))

    history = dict(history)
    history["test_metrics"] = test_metrics
    return history
