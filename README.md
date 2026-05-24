# Ark+ 3D: Heterogeneous Label Learning on MedMNIST 3D

## Abstract

This repository adapts **Ark+** (Ma et al., *Nature* 2025) — a student-teacher cyclic pretraining framework with EMA and heterogeneous label-space consistency — to the six **MedMNIST 3D** benchmark datasets. The key architectural innovation is replacing 3D convolutions with a **2D Swin-Tiny-per-slice + Depth Transformer** encoder, enabling use of ImageNet-pretrained weights while staying within 8 GB VRAM on an RTX 4060.

---

## Architecture

```
Input (B, 1, D, 28, 28)
  │
  ├─ Upsample slices → (B×D, 3, 224, 224)
  │
  ├─ 2D Swin-Tiny (ImageNet pretrained)
  │     └─ per-slice feature → (B×D, 768)
  │
  ├─ SliceProjector MLP → (B, D, 768)
  │
  ├─ DepthAggregator Transformer (2 layers, 8 heads)
  │     └─ learnable depth positional encoding
  │     └─ global avg pool → (B, 768)
  │
  ├─ Student/Teacher Projector MLP
  │     768 → 4096 → 1376  [consistency loss target]
  │
  └─ Task Heads (one per dataset)
        768 → C_i  [BCE or CE task loss]
```

**Student-Teacher EMA**: teacher weights are updated as `θ_t ← m·θ_t + (1-m)·θ_s` with `m` cosine-annealed from 0.996 → 1.0 over training.

**Loss formulation**:
```
total_loss = (1 - coff) × task_loss + coff × MSE(s_proj, t_proj)
coff = min(0.5, (m - 0.9) × 5)
```

Task losses are normalized by `1/ln(n_classes)` for equal gradient scale across datasets.

---

## Installation

```bash
# 1. Create conda environment
conda create -n ark python=3.10
conda activate ark

# 2. Install PyTorch with CUDA (RTX 4060 → CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt
```

---

## Quick Start

Open `ark_plus_3d/notebooks/ark_3d_run.ipynb` in JupyterLab and run cells 1–6 in order.

```bash
cd ark_plus_3d/notebooks
jupyter notebook ark_3d_run.ipynb
```

| Cell | Action |
|------|--------|
| 1    | Import modules |
| 2    | Configure datasets, hyperparameters, device |
| 3    | Build DataLoaders with class-balanced samplers |
| 4    | Instantiate ArkPlus3D, check forward pass |
| 5    | Train (cyclic multi-task loop) |
| 6    | Plot training curves + final test summary |

---

## SOTA Benchmarks (Ark+, Nature 2025)

| Dataset        | Metric | SOTA  |
|----------------|--------|-------|
| OrganMNIST3D   | ACC    | 0.997 |
| NoduleMNIST3D  | AUC    | 0.863 |
| AdrenalMNIST3D | AUC    | 0.874 |
| FractureMNIST3D| ACC    | 0.714 |
| VesselMNIST3D  | AUC    | 0.914 |
| SynapseMNIST3D | AUC    | 0.843 |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 6 GB    | 8 GB (RTX 4060) |
| RAM       | 12 GB   | 16 GB |
| Storage   | 2 GB (datasets + checkpoints) | — |

If you encounter OOM errors, reduce `BATCH_SIZE` from 8 to 4 in Cell 2.

---

## Project Structure

```
ark_plus_3d/
├── models/
│   ├── swin3d.py       # 3D encoder: 2D Swin per slice + Depth Transformer
│   └── ark_plus.py     # Student-teacher EMA model + task heads
├── data/
│   └── dataset.py      # MedMNIST 3D loader + dual augmentations
├── training/
│   └── engine.py       # Cyclic training loop + evaluation
├── utils/
│   ├── metrics.py      # AUROC, ACC, CSV logging, plotting
│   └── helpers.py      # LR schedule, checkpointing, seeding
├── configs/
│   └── medmnist_3d.yaml
└── notebooks/
    └── ark_3d_run.ipynb
docs/
    ├── README.md
    └── METHODOLOGY.md
requirements.txt
```

---

## Citation

```bibtex
@article{ma2025ark,
  title={A fully open AI foundation model for chest X-rays (Ark+)},
  author={Ma, Jun and others},
  journal={Nature},
  year={2025}
}
```
