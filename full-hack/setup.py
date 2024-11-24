from setuptools import setup, find_packages

setup(
    name="retail-product-segmentation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A pipeline for retail product segmentation and semantic grouping using FastSAM",
    url="https://github.com/yourusername/retail-product-segmentation",
    packages=find_packages(),  # Automatically find Python packages
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.3",
        "matplotlib>=3.4.3",
        "fastsam>=0.1.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
