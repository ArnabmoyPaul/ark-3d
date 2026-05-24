"""
swin3d.py — Memory-efficient 3D encoder using pretrained 2D Swin-Tiny.
Architecture:
  Input (B, 1, D, H, W)
  → For each depth slice: grayscale→RGB → 2D Swin → 768-dim feature
  → MLP projection + LayerNorm
  → DepthAggregator: Transformer over D positions + learnable pos enc
  → Global avg pool over depth → LayerNorm → 768-dim volume embedding
VRAM budget: ~4–5 GB at batch_size=8 on RTX 4060 (8 GB).
"""

import torch
import torch.nn as nn
import timm
from einops import rearrange


class SliceProjector(nn.Module):
    """Projects 2D Swin CLS token (768) → 768-dim with MLP + LayerNorm + GELU."""

    def __init__(self, in_dim: int = 768, out_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthAggregator(nn.Module):
    """
    Transformer encoder over depth positions with learnable positional encoding.
    Input:  (B, D, C)  — one feature per slice
    Output: (B, C)     — globally pooled volume embedding
    """

    def __init__(self, embed_dim: int = 768, max_depth: int = 64,
                 nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_depth, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,  # 3072 for embed_dim=768
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, C)
        Returns:
            (B, C)
        """
        B, D, C = x.shape
        pos = self.pos_embed[:, :D, :]          # (1, D, C)
        x = x + pos                              # broadcast over batch
        x = self.transformer(x)                  # (B, D, C)
        x = x.mean(dim=1)                        # global avg pool over depth
        return self.norm(x)                      # (B, C)


class Swin3DEncoder(nn.Module):
    """
    Memory-efficient 3D encoder:
      1. Pretrained 2D Swin-Tiny processes each depth slice independently.
      2. Slice features are aggregated by a depth Transformer.

    Input shape:  (B, 1, D, H, W)  — single-channel 3D volume
    Output shape: (B, 768)          — volume-level embedding

    Memory strategy:
      - Swin operates on (B*D, 3, H, W) — slices are batched along B*D dim.
      - For batch_size=8 and D≤28, this gives B*D≤224 forward passes,
        which is handled efficiently by Swin's patch embedding.
      - H=W=28 is the MedMNIST 3D native resolution (no upsample needed
        since swin_tiny uses adaptive pool for feature extraction).
    """

    def __init__(
        self,
        swin_model: str = "swin_tiny_patch4_window7_224",
        embed_dim: int = 768,
        max_depth: int = 64,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── 2D Swin backbone ──────────────────────────────────────────────
        # num_classes=0 → no classification head, returns (B, C) after global pool
        self.swin2d = timm.create_model(
            swin_model,
            pretrained=pretrained,
            num_classes=0,          # removes head, returns pooled features
            global_pool="avg",
        )
        swin_out_dim = self.swin2d.num_features  # 768 for swin_tiny

        # ── Slice feature projector ───────────────────────────────────────
        self.slice_projector = SliceProjector(in_dim=swin_out_dim, out_dim=embed_dim)

        # ── Depth aggregator ─────────────────────────────────────────────
        self.depth_agg = DepthAggregator(
            embed_dim=embed_dim,
            max_depth=max_depth,
            nhead=8,
            num_layers=num_transformer_layers,
            dropout=dropout,
        )

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W)  — grayscale 3D volume
        Returns:
            (B, embed_dim)
        """
        B, C, D, H, W = x.shape
        assert C == 1, f"Expected 1-channel input, got {C}"

        # Flatten batch & depth: (B, 1, D, H, W) → (B*D, 1, H, W)
        x = rearrange(x, "b c d h w -> (b d) c h w")

        # Grayscale → RGB by repeating channel: (B*D, 3, H, W)
        x = x.repeat(1, 3, 1, 1)

        # MedMNIST 3D: H=W=28, but swin_tiny expects ≥224 for full window coverage.
        # Upsample to 224 for full pretrained weight utilisation.
        if H != 224 or W != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                          align_corners=False)

        # 2D Swin forward — (B*D, swin_out_dim)
        slice_feats = self.swin2d(x)

        # Project slices: (B*D, embed_dim)
        slice_feats = self.slice_projector(slice_feats)

        # Restore depth dim: (B, D, embed_dim)
        slice_feats = rearrange(slice_feats, "(b d) c -> b d c", b=B, d=D)

        # Depth aggregation → (B, embed_dim)
        volume_embed = self.depth_agg(slice_feats)

        return volume_embed


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Swin3DEncoder(pretrained=False).to(device)
    x = torch.randn(2, 1, 28, 28, 28, device=device)
    out = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")   # Expected: (2, 768)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {params:.1f}M")
