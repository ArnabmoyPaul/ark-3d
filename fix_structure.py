"""
Run this script ONCE from D:\ark+3d to create the correct package structure.
In your conda terminal (ark env activated):
    cd D:\ark+3d
    python fix_structure.py
"""
import os, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))  # D:\ark+3d

# Folders to create
dirs = [
    "ark_plus_3d",
    "ark_plus_3d/models",
    "ark_plus_3d/data",
    "ark_plus_3d/training",
    "ark_plus_3d/utils",
    "ark_plus_3d/configs",
    "ark_plus_3d/notebooks",
    "checkpoints",
    "data",
]
for d in dirs:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    print(f"  mkdir: {d}")

# Create __init__.py in every package folder
init_dirs = [
    "ark_plus_3d",
    "ark_plus_3d/models",
    "ark_plus_3d/data",
    "ark_plus_3d/training",
    "ark_plus_3d/utils",
    "ark_plus_3d/configs",
]
for d in init_dirs:
    p = os.path.join(ROOT, d, "__init__.py")
    if not os.path.exists(p):
        open(p, "w").close()
        print(f"  created: {d}/__init__.py")

# Move files to correct locations
moves = {
    "ark_plus.py":      "ark_plus_3d/models/ark_plus.py",
    "swin3d.py":        "ark_plus_3d/models/swin3d.py",
    "dataset.py":       "ark_plus_3d/data/dataset.py",
    "engine.py":        "ark_plus_3d/training/engine.py",
    "metrics.py":       "ark_plus_3d/utils/metrics.py",
    "helpers.py":       "ark_plus_3d/utils/helpers.py",
    "medmnist_3d.yaml": "ark_plus_3d/configs/medmnist_3d.yaml",
    "ark_3d_run.ipynb": "ark_plus_3d/notebooks/ark_3d_run.ipynb",
}

for src_name, dst_rel in moves.items():
    src = os.path.join(ROOT, src_name)
    dst = os.path.join(ROOT, dst_rel)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
        print(f"  moved: {src_name} → {dst_rel}")
    elif os.path.exists(dst):
        print(f"  already OK: {dst_rel}")
    else:
        print(f"  WARNING: source not found: {src_name}")

print("\nDone! Structure:")
for root, dirs_list, files in os.walk(os.path.join(ROOT, "ark_plus_3d")):
    level = root.replace(ROOT, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        print(f"{indent}  {f}")
