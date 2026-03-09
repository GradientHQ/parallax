## Parallax MLX Kernel Extentions
Extended kernels built for MLX backend.
MLX official instructions for custom extensions: https://ml-explore.github.io/mlx/build/html/dev/extensions.html

### Directory Structure
```bash
.
├── __init__.py
├── bindings.cpp                   # Nanobind
├── CMakelists.txt
├── lib
│   ├── _ext.cpython-311-darwin.so # Python 3.11 binding
│   ├── _ext.cpython-312-darwin.so # Python 3.12 binding
│   ├── _ext.cpython-313-darwin.so # Python 3.13 binding
│   ├── libparallax_ext.dylib      # C++ extension library
│   └── parallax_ext.metallib      # Metal library
├── paged_attention_v1             # Kernel Source Code Directories
│   ├── float8.metal
│   ├── paged_attention.cpp
│   ├── paged_attention.h
│   ├── paged_attention.metal
│   ├── reshape_and_cache.metal
│   └── utils.metal
├── README.md
└── setup.py                       # Setup Tools Script
```

### Package Build and Install
Build inplace for development using:
```sh
python setup.py build_ext -j8 --inplace
```
Then you can try to install in the directory using the command ```python -m pip install .```.
The pre-built package should be already installed in the parallax project.

When multiple prebuilt `_ext.cpython-<ver>-darwin.so` files are present in `lib/`,
Parallax automatically loads the one matching the current Python runtime.

### Usage Example
```python
import mlx.core as mx
from parallax_extensions.ops import paged_attention_v1
```
