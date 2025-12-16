## Parallax MLX Kernel Extentions
Extended kernels built for MLX backend.
MLX official instructions for custom extensions: https://ml-explore.github.io/mlx/build/html/dev/extensions.html

### Directory Structure
.
├── bindings.cpp          # Nanobind
├── CMakelists.txt
├── paged_attention_v1    # Kernel Source Code Directories
│   ├── float8.metal
│   ├── paged_attention.cpp
│   ├── paged_attention.h
│   ├── paged_attention.metal
│   ├── reshape_and_cache.metal
│   └── utils.metal
├── parallax_extensions
│   ├── __init__.py
│   ├── _ext.cpython-311-darwin.so # Python Binding
│   ├── libparallax_ext.dylib      # C++ extension library
│   └── parallax_ext.metallib      # Metal library
├── README.md
└── setup.py                       # Build Script

### Package Build and Install
```sh
python setup.py build_ext -j8 --inplace

pip install -e .
```
Or use pre-built package by installing the main parallax package.

### Usage Example
```python
import mlx.core as mx
from parallax_extensions import paged_attention_v1
```
