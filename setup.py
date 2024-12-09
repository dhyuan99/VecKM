from setuptools import setup, find_packages

setup(
    name="VecKM",
    version="0.1.0",
    author="Dehao Yuan",
    author_email="dhyuan@umd.edu",
    description="A very efficient and descriptive local geometry encoder / point tokenizer / patch embedder.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dhyuan99/VecKM",  # Replace with your repo
    packages=find_packages(),  # Automatically finds sub-packages
    install_requires=[
        "torch>=1.13.0",
        "scipy==1.14.1",
        "scikit-learn==1.5.0",
        "tqdm==4.66.2",
        "matplotlib==3.8.3",
        "matplotlib-inline==0.1.7",
        "pillow==11.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)