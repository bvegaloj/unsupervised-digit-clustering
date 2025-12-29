from setuptools import setup, find_packages

setup(
    name="unsupervised-digit-clustering",
    version="0.1.0",
    description="Unsupervised clustering algorithms applied to MNIST digit dataset",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "tensorflow>=2.12.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "pytest>=6.0.0",
    ],
    python_requires=">=3.8",
)
