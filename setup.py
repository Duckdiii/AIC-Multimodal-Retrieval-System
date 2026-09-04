from setuptools import setup, find_packages

setup(
    name="aic_core",
    version="4.1.0",
    description="Unified Online Video Retrieval Core for AIC Competition",
    packages=find_packages(include=["aic_core", "aic_core.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "faiss-cpu",
        "torch",
        "transformers",
    ],
)
