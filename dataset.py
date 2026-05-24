"""
dataset.py — MedMNIST 3D dataset with Ark+ dual-view augmentation.

6 MedMNIST 3D datasets:
  organmnist3d   — 11-class,  ACC metric
  nodulemnist3d  — binary,    AUC metric
  adrenalmnist3d — binary,    AUC metric
  fracturemnist3d— 3-class,   ACC metric
  vesselmnist3d  — binary,    AUC metric
  synapsemnist3d — binary,    AUC metric

Augmentation policy (Ark+ §Data augmentation):
  Student (strong): flips, rotations, intensity scaling, brightness, noise, blur
  Teacher (weak)  : depth flip + mild intensity only
  Val/Test        : normalize to [-1, 1]
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import medmnist
from medmnist import INFO


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "organmnist3d": {
        "flag": "organmnist3d",
        "task": "multi-class",
        "n_classes": 11,
        "metric": "ACC",
    },
    "nodulemnist3d": {
        "flag": "nodulemnist3d",
        "task": "binary",
        "n_classes": 2,
        "metric": "AUC",
    },
    "adrenalmnist3d": {
        "flag": "adrenalmnist3d",
        "task": "binary",
        "n_classes": 2,
        "metric": "AUC",
    },
    "fracturemnist3d": {
        "flag": "fracturemnist3d",
        "task": "multi-class",
        "n_classes": 3,
        "metric": "ACC",
    },
    "vesselmnist3d": {
        "flag": "vesselmnist3d",
        "task": "binary",
        "n_classes": 2,
        "metric": "AUC",
    },
    "synapsemnist3d": {
        "flag": "synapsemnist3d",
        "task": "binary",
        "n_classes": 2,
        "metric": "AUC",
    },
}

# Friendly name → flag mapping (used in notebook)
NAME_TO_FLAG = {
    "OrganMNIST3D":   "organmnist3d",
    "NoduleMNIST3D":  "nodulemnist3d",
    "AdrenalMNIST3D": "adrenalmnist3d",
    "FractureMNIST3D":"fracturemnist3d",
    "VesselMNIST3D":  "vesselmnist3d",
    "SynapseMNIST3D": "synapsemnist3d",
}


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers (pure numpy, applied on (D, H, W) arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _random_flip(volume: np.ndarray, axis: int, p: float = 0.5) -> np.ndarray:
    if random.random() < p:
        volume = np.flip(volume, axis=axis).copy()
    return volume


def _random_rotate90(volume: np.ndarray, axes=(1, 2), p: float = 0.5) -> np.ndarray:
    if random.random() < p:
        k = random.choice([1, 2, 3])
        volume = np.rot90(volume, k=k, axes=axes).copy()
    return volume


def _intensity_scale(volume: np.ndarray, low: float = 0.7,
                     high: float = 1.3, p: float = 0.5) -> np.ndarray:
    if random.random() < p:
        scale = random.uniform(low, high)
        volume = volume * scale
    return volume


def _brightness_shift(volume: np.ndarray, max_shift: float = 0.1,
                      p: float = 0.3) -> np.ndarray:
    if random.random() < p:
        shift = random.uniform(-max_shift, max_shift)
        volume = volume + shift
    return volume


def _gaussian_noise(volume: np.ndarray, sigma: float = 0.02,
                    p: float = 0.3) -> np.ndarray:
    if random.random() < p:
        noise = np.random.normal(0, sigma, volume.shape).astype(np.float32)
        volume = volume + noise
    return volume


def _uniform_blur(volume: np.ndarray, p: float = 0.2) -> np.ndarray:
    """Simple uniform blur via box filter approximation."""
    if random.random() < p:
        from scipy.ndimage import uniform_filter
        sigma = random.uniform(0.5, 1.0)
        volume = uniform_filter(volume, size=max(1, int(sigma * 2 + 1))).astype(np.float32)
    return volume


def _normalize(volume: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1] based on min-max of the volume."""
    v_min, v_max = volume.min(), volume.max()
    if v_max - v_min > 1e-6:
        volume = 2.0 * (volume - v_min) / (v_max - v_min) - 1.0
    return volume.astype(np.float32)


def augment_student(volume: np.ndarray) -> np.ndarray:
    """Strong augmentation for student view."""
    # Random axis flips
    for axis in range(3):  # D, H, W axes of (D, H, W)
        volume = _random_flip(volume, axis=axis, p=0.5)
    # 90° rotation in H-W plane
    volume = _random_rotate90(volume, axes=(1, 2), p=0.5)
    # Intensity scaling
    volume = _intensity_scale(volume, low=0.7, high=1.3, p=0.5)
    # Brightness shift
    volume = _brightness_shift(volume, max_shift=0.1, p=0.3)
    # Gaussian noise
    volume = _gaussian_noise(volume, sigma=0.02, p=0.3)
    # Uniform blur
    volume = _uniform_blur(volume, p=0.2)
    return _normalize(volume)


def augment_teacher(volume: np.ndarray) -> np.ndarray:
    """Weak augmentation for teacher view."""
    # Depth flip only
    volume = _random_flip(volume, axis=0, p=0.3)
    # Mild intensity
    volume = _intensity_scale(volume, low=0.95, high=1.05, p=0.2)
    return _normalize(volume)


def augment_val(volume: np.ndarray) -> np.ndarray:
    """No augmentation for val/test."""
    return _normalize(volume)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class MedMNIST3DDataset(Dataset):
    """
    Wraps a MedMNIST 3D dataset and returns dual views for Ark+ training.

    Returns:
        split='train':
            (student_view, teacher_view, label)
            student_view : (1, D, H, W) float32
            teacher_view : (1, D, H, W) float32
            label        : float32 (1,) for binary | long scalar for multi-class

        split='val'/'test':
            (volume, volume, label)   — same view returned twice for API consistency
    """

    def __init__(self, flag: str, split: str = "train",
                 download: bool = True, data_root: str = "./data"):
        super().__init__()
        assert flag in DATASET_CONFIGS, f"Unknown flag: {flag}"
        self.cfg = DATASET_CONFIGS[flag]
        self.split = split
        self.is_binary = (self.cfg["task"] == "binary")

        # Load medmnist dataset
        DataClass = getattr(medmnist, INFO[flag]["python_class"])
        self.dataset = DataClass(
            split=split,
            download=download,
            root=data_root,
            size=28,   # 28³ native resolution
        )

        # Cache images and labels as numpy for fast access
        self.images = self.dataset.imgs          # (N, D, H, W) uint8
        self.labels = self.dataset.labels        # (N, 1) int

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        # uint8 [0,255] → float32 [0,1]
        volume = self.images[idx].astype(np.float32) / 255.0  # (D, H, W)

        raw_label = int(self.labels[idx].item())

        if self.split == "train":
            s_view = augment_student(volume.copy())
            t_view = augment_teacher(volume.copy())
        else:
            s_view = augment_val(volume.copy())
            t_view = s_view  # identical for val/test

        # (D, H, W) → (1, D, H, W)
        s_tensor = torch.from_numpy(s_view).unsqueeze(0)
        t_tensor = torch.from_numpy(t_view).unsqueeze(0)

        # Label dtype: float32 scalar (1,) for binary BCE, long scalar for CE
        if self.is_binary:
            label = torch.tensor([raw_label], dtype=torch.float32)
        else:
            label = torch.tensor(raw_label, dtype=torch.long)

        return s_tensor, t_tensor, label

    @property
    def class_labels(self) -> np.ndarray:
        """Flat array of integer class indices — used by WeightedRandomSampler."""
        return self.labels.flatten().astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(flag: str, data_root: str = "./data"):
    """
    Build train / val / test splits for a given MedMNIST 3D flag.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    train = MedMNIST3DDataset(flag, split="train", data_root=data_root)
    val   = MedMNIST3DDataset(flag, split="val",   data_root=data_root)
    test  = MedMNIST3DDataset(flag, split="test",  data_root=data_root)
    return train, val, test


def get_weighted_sampler(dataset: MedMNIST3DDataset) -> WeightedRandomSampler:
    """
    Compute per-sample weights for class-balanced sampling.
    Critical for imbalanced datasets (e.g. NoduleMNIST3D, SynapseMNIST3D).
    """
    labels = dataset.class_labels                     # (N,)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


if __name__ == "__main__":
    # Quick smoke test
    train, val, test = build_datasets("nodulemnist3d")
    print(f"NoduleMNIST3D — train: {len(train)}, val: {len(val)}, test: {len(test)}")

    s, t, lbl = train[0]
    print(f"Student view shape : {tuple(s.shape)}")
    print(f"Teacher view shape : {tuple(t.shape)}")
    print(f"Label              : {lbl} (dtype={lbl.dtype})")

    sampler = get_weighted_sampler(train)
    loader = DataLoader(train, batch_size=4, sampler=sampler, num_workers=0)
    batch_s, batch_t, batch_lbl = next(iter(loader))
    print(f"\nBatch student: {tuple(batch_s.shape)}")
    print(f"Batch teacher: {tuple(batch_t.shape)}")
    print(f"Batch labels : {tuple(batch_lbl.shape)}")
