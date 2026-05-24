from setuptools import setup, find_packages

setup(
    name="ark_plus_3d",
    version="0.1.0",
    description="Ark+ student-teacher cyclic pretraining on MedMNIST 3D datasets",
    author="Arnabmoy Paul",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "medmnist>=3.0.0",
        "einops>=0.6.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "PyYAML>=6.0",
        "matplotlib>=3.5.0",
        "scipy>=1.9.0",
        "tqdm>=4.64.0",
    ],
)
