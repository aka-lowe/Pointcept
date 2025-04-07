import subprocess
import sys
import os
import setuptools


with open("requirements.txt") as f:
    required = f.read().splitlines()



# install local dependencies
local_packages = [
    './libs/pointops',
    './libs/pointgroup_ops'
]

setuptools.setup(
    name="Pointcept",
    version="0.0.1",
    author="Pointcept",
    packages=setuptools.find_packages(),
    install_requires=required,
    python_requires="==3.12",
    dependency_links=[
        'git+https://github.com/octree-nn/ocnn-pytorch.git,'
        'git+https://github.com/openai/CLIP.git',
        'git+https://github.com/Dao-AILab/flash-attention.git',
        'git+https://github.com/Silverster98/pointops.git'
    ]
)