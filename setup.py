from setuptools import find_packages, setup

setup(
    name="quest",
    version="0.1.0",
    description="Query-based virtual staining: multiplex immunofluorescence predicted from H&E",
    python_requires=">=3.10,<3.11",
    packages=find_packages(include=["quest", "quest.*", "questkit", "questkit.*"]),
    install_requires=[
        "torch>=2.0",
        "timm>=0.9.16",
        "einops>=0.6",
        "numpy>=1.24,<2",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "scikit-image>=0.21",
        "matplotlib>=3.7",
        "h5py>=3.8",
        "huggingface_hub>=0.20",
        "omegaconf>=2.3",
        "jupyter>=1.0",
        "deepcell>=0.12",
    ],
)
