# Methodology: Ark+ 3D for MedMNIST

## 1. Why 2D Swin + Depth Attention Instead of 3D CNN

### Memory argument

A standard 3D ResNet-50 processing a `(B=8, 1, 28, 28, 28)` batch accumulates feature maps through all spatial dimensions simultaneously. On an RTX 4060 (8 GB), a 3D Swin-Small at native resolution would require roughly 10–12 GB just for activations during the forward pass — exceeding the hardware budget.

The slice-wise approach decouples the spatial and depth dimensions:

```
3D CNN path   : B × D × H × W activations simultaneously  → OOM at B=8
2D Swin path  : (B×D) slices, processed independently     → ~4–5 GB at B=8
```

Depth is reintroduced by a lightweight Transformer Encoder (2 layers, 8 heads) *after* feature extraction, which adds only ~5 M parameters and negligible memory.

### Pretrained weight argument

No large-scale 3D medical imaging pretraining corpus is publicly available in the general-purpose sense. ImageNet-pretrained 2D Swin-Tiny has learned rich hierarchical visual features (edges, textures, shapes) that transfer well to 2D slices of CT/microscopy volumes. Initialising from ImageNet weights gives a ~5–10% accuracy boost over random init at epoch 10 on MedMNIST 3D benchmarks.

---

## 2. Student-Teacher EMA Mechanism

The teacher is a time-averaged (exponential moving average) copy of the student:

```
θ_teacher ← m × θ_teacher + (1 - m) × θ_student
```

This produces a "smoother" and more stable feature extractor than the instantaneous student. The consistency loss:

```
L_cons = MSE(student_proj, teacher_proj.detach())
```

forces the student to produce similar representations to the teacher under strong augmentation (student view) vs. weak augmentation (teacher view). This acts as a form of self-supervised regularisation even when labelled data is limited.

**Momentum schedule**: `m` is cosine-annealed from `m_base=0.996` to `1.0` over training:

```
m(epoch) = 1 - (1 - m_base) × (cos(π × epoch/T) + 1) / 2
```

Early in training, `m` is smaller (0.996), so the teacher updates quickly and learns alongside the student. Late in training, `m → 1.0`, effectively freezing the teacher as a stable target — preventing collapse.

**Consistency coefficient**:
```
coff = min(0.5, (m - 0.9) × 5)
```

When `m = 0.996`: `coff = min(0.5, 0.048) = 0.048` — task loss dominates.
When `m = 0.999`: `coff = min(0.5, 0.495) = 0.495` — balanced task + consistency.
When `m = 1.000`: `coff = 0.5` — maximum consistency weight.

This naturally ramps consistency loss as the teacher becomes more reliable.

---

## 3. Loss Normalisation by ln(n_classes)

When training on datasets of different task sizes simultaneously, gradients from 11-class OrganMNIST3D will dominate those from binary NoduleMNIST3D unless explicitly balanced.

**Entropy argument**: for a uniform categorical distribution over `C` classes, the maximum cross-entropy is `ln(C)`. Weighting each task by `1/ln(C)` normalises the loss to the same scale regardless of label space size:

| Dataset         | Classes | ln(C) | Weight |
|-----------------|---------|-------|--------|
| OrganMNIST3D    | 11      | 2.398 | 0.417  |
| FractureMNIST3D | 3       | 1.099 | 0.910  |
| Binary tasks    | 2       | 0.693 | 1.443  |

Binary tasks receive the highest weight — compensating for their lower theoretical entropy ceiling.

---

## 4. Cyclic vs Simultaneous Multi-Dataset Training

**Simultaneous**: all datasets contribute to one combined gradient per step. This requires equal batch sizes and identical loss scales — hard to achieve with heterogeneous datasets of varying sizes and class counts.

**Cyclic (Ark+ approach)**: iterate round-robin over datasets, computing one task's gradient at a time. Benefits:
- Works naturally with datasets of different sizes (iterators cycle independently)
- Class-imbalance is handled per-dataset via `WeightedRandomSampler`
- Gradient interference between tasks is reduced
- Memory footprint is bounded by the largest single-dataset batch, not their sum

On small datasets (MedMNIST 3D typically has 1K–5K training samples), cyclic training ensures each dataset sees enough gradient updates per epoch even when others are much larger.

---

## 5. Expected Results

### Quick test (10 epochs, batch_size=8, 3 datasets)

| Dataset        | Metric | Expected |
|----------------|--------|----------|
| OrganMNIST3D   | ACC    | 0.65–0.75 |
| NoduleMNIST3D  | AUC    | 0.70–0.78 |
| AdrenalMNIST3D | AUC    | 0.72–0.80 |

At epoch 10, training is still in the warmup/early phase. Pretrained Swin weights give a strong start.

### Mid training (50 epochs)

| Dataset        | Metric | Expected |
|----------------|--------|----------|
| OrganMNIST3D   | ACC    | 0.82–0.88 |
| NoduleMNIST3D  | AUC    | 0.80–0.85 |
| AdrenalMNIST3D | AUC    | 0.82–0.86 |

Consistency loss is now active and the teacher has stabilised. Cyclic training has exposed the model to the full dataset diversity.

### Converged (200 epochs)

| Dataset        | Metric | Expected | SOTA  |
|----------------|--------|----------|-------|
| OrganMNIST3D   | ACC    | 0.93–0.96 | 0.997 |
| NoduleMNIST3D  | AUC    | 0.84–0.87 | 0.863 |
| AdrenalMNIST3D | AUC    | 0.86–0.90 | 0.874 |
| FractureMNIST3D| ACC    | 0.67–0.72 | 0.714 |
| VesselMNIST3D  | AUC    | 0.88–0.92 | 0.914 |
| SynapseMNIST3D | AUC    | 0.82–0.86 | 0.843 |

Expected to match or slightly exceed SOTA on binary tasks due to consistency regularisation. OrganMNIST3D gap is partly due to using Swin-Tiny (28M) vs Swin-Large (307M) in the original Ark+.

---

## 6. Hyperparameter Sensitivity

| Hyperparameter   | Recommended | Notes |
|------------------|-------------|-------|
| `lr`             | 1e-4        | Higher LR (1e-3) causes instability with transformers |
| `warmup_epochs`  | 5–10        | Shorter warmup → erratic early gradients |
| `momentum`       | 0.996       | Lower (0.99) → teacher too noisy; higher (0.999) → slow adaptation |
| `batch_size`     | 8           | Minimum 4; BN in projector requires >1 per batch |
| `weight_decay`   | 0.05        | AdamW default; prevents projector weight explosion |
| `grad_clip`      | 1.0         | Essential for transformer stability; do not remove |
