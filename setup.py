from setuptools import setup, find_packages

setup(
    name="legal-clusterer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.3",
        "sentence-transformers>=2.2",
        "transformers>=4.40",
        "numpy>=1.24",
        "scipy>=1.10",
        "tqdm>=4.65",
        "pypdf>=6.60",
        "python-docx>=1.1",
        "accelerate>=0.26",
    ],
)