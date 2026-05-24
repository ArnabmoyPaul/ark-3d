"""
ark_plus.py — Full Ark+ student-teacher model (Ma et al., Nature 2025).

Key components:
  - Student Encoder   : Swin3DEncoder (trainable)
  - Teacher Encoder   : Swin3DEncoder (EMA-updated, frozen gradients)
  - Student/Teacher Projectors : MLP 768 → 4096 → 1376
  - Task Heads        : one Linear head per dataset

EMA update rule (Ark+ §Methods):
  θ_t ← m·θ_t + (1-m)·θ_s
where m increases from momentum_init → 1.0 via cosine schedule.
"""

import copy
import math
import torch
import torch.nn as nn

from ark_plus_3d.models.swin3d import Swin3DEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Projector MLP
# ─────────────────────────────────────────────────────────────────────────────

class ProjectorMLP(nn.Module):
    """
    3-layer bottleneck projector:
      Linear(768 → 4096) → BN → GELU → Dropout → Linear(4096 → proj_dim)
    Matches Ark+ projector design for consistency loss.
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 4096,
                 out_dim: int = 1376, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
        # Ark+ init: weight ~ N(0, 0.01), bias = 0 on output layer
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Task Head
# ─────────────────────────────────────────────────────────────────────────────

class TaskHead(nn.Module):
    """Single linear classification head."""

    def __init__(self, embed_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
# ArkPlus3D — main model
# ─────────────────────────────────────────────────────────────────────────────

class ArkPlus3D(nn.Module):
    """
    Ark+ adapted for 3D MedMNIST datasets.

    Args:
        num_classes_list : list of int — number of classes per dataset
        swin_model       : timm model name for 2D backbone
        embed_dim        : encoder output dimension (768 for swin_tiny)
        projector_dim    : consistency projector output dim (1376 per Ark+)
        pretrained       : load ImageNet weights for Swin backbone
        dropout          : dropout rate in projector & depth aggregator

    Usage:
        model = ArkPlus3D(num_classes_list=[11, 2, 2])
        s_proj, s_logits = model(x, head_idx=0, mode='student')
        with torch.no_grad():
            t_proj, _ = model(x, head_idx=0, mode='teacher')
    """

    def __init__(
        self,
        num_classes_list: list,
        swin_model: str = "swin_tiny_patch4_window7_224",
        embed_dim: int = 768,
        projector_dim: int = 1376,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.projector_dim = projector_dim
        self.num_datasets = len(num_classes_list)

        # ── Student components (trainable) ────────────────────────────────
        self.student_encoder = Swin3DEncoder(
            swin_model=swin_model,
            embed_dim=embed_dim,
            pretrained=pretrained,
            dropout=dropout,
        )
        self.student_projector = ProjectorMLP(
            in_dim=embed_dim,
            hidden_dim=4096,
            out_dim=projector_dim,
            dropout=dropout,
        )

        # ── Task heads (one per dataset) ──────────────────────────────────
        self.task_heads = nn.ModuleList([
            TaskHead(embed_dim=embed_dim, num_classes=nc)
            for nc in num_classes_list
        ])

        # ── Teacher components (EMA, no gradients) ────────────────────────
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_projector = copy.deepcopy(self.student_projector)

        # Freeze teacher
        for p in self.teacher_encoder.parameters():
            p.requires_grad_(False)
        for p in self.teacher_projector.parameters():
            p.requires_grad_(False)

    # ── EMA update ────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """
        EMA update: θ_t ← m·θ_t + (1-m)·θ_s
        Call AFTER optimizer.step() every batch.
        """
        for s_param, t_param in zip(
            self.student_encoder.parameters(),
            self.teacher_encoder.parameters()
        ):
            t_param.data.mul_(momentum).add_(s_param.data * (1.0 - momentum))

        for s_param, t_param in zip(
            self.student_projector.parameters(),
            self.teacher_projector.parameters()
        ):
            t_param.data.mul_(momentum).add_(s_param.data * (1.0 - momentum))

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        head_idx: int,
        mode: str = "student",
    ):
        """
        Args:
            x        : (B, 1, D, H, W) — 3D volume
            head_idx : which task head to use
            mode     : 'student' (trainable) | 'teacher' (EMA, use no_grad externally)

        Returns:
            (proj, logits)
              proj   : (B, projector_dim)
              logits : (B, num_classes[head_idx])  — None for teacher
        """
        if mode == "student":
            feat = self.student_encoder(x)           # (B, embed_dim)
            proj = self.student_projector(feat)       # (B, projector_dim)
            logits = self.task_heads[head_idx](feat)  # (B, num_classes)
            return proj, logits

        elif mode == "teacher":
            feat = self.teacher_encoder(x)            # (B, embed_dim)
            proj = self.teacher_projector(feat)       # (B, projector_dim)
            return proj, None

        else:
            raise ValueError(f"mode must be 'student' or 'teacher', got '{mode}'")

    # ── Utility ───────────────────────────────────────────────────────────

    def trainable_parameters(self):
        """Return only student + task head parameters for optimizer."""
        return (
            list(self.student_encoder.parameters())
            + list(self.student_projector.parameters())
            + list(self.task_heads.parameters())
        )

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_M": total / 1e6, "trainable_M": trainable / 1e6}

    @staticmethod
    def cosine_momentum(epoch: int, total_epochs: int,
                        m_base: float = 0.996) -> float:
        """
        Cosine schedule: m_base → 1.0 over training.
        Ark+ §Methods: momentum cosine annealing.
        """
        return 1.0 - (1.0 - m_base) * (
            math.cos(math.pi * epoch / total_epochs) + 1.0
        ) / 2.0


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ArkPlus3D(
        num_classes_list=[11, 2, 2],
        pretrained=False,
    ).to(device)

    counts = model.param_count()
    print(f"Total params    : {counts['total_M']:.1f} M")
    print(f"Trainable params: {counts['trainable_M']:.1f} M")

    x = torch.randn(2, 1, 28, 28, 28, device=device)

    # Student forward
    proj, logits = model(x, head_idx=0, mode="student")
    print(f"\nStudent proj  : {tuple(proj.shape)}")    # (2, 1376)
    print(f"Student logits: {tuple(logits.shape)}")   # (2, 11)

    # Teacher forward
    with torch.no_grad():
        t_proj, _ = model(x, head_idx=0, mode="teacher")
    print(f"Teacher proj  : {tuple(t_proj.shape)}")   # (2, 1376)

    # EMA update test
    model.update_teacher(momentum=0.996)
    print("\nEMA update: OK")
