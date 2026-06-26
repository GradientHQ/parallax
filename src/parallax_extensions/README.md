## Parallax MLX Kernel Extensions
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
├── kernels                         # Kernel source code directories
│   ├── common
│   │   ├── float8.metal
│   │   ├── utils.cpp
│   │   ├── utils.h
│   │   └── utils.metal
│   ├── dsa
│   ├── indexer_cache
│   ├── mla
│   ├── msa
│   ├── paged_attention
│   └── reshape_and_cache
├── README.md
└── setup.py                       # Setup Tools Script
```

### Package Build and Install
Build inplace for development using:
```sh
cd src/parallax_extensions
../../.venv/bin/python setup.py build_ext -j8 --inplace
```
Then you can try to install the package using
```sh
../../.venv/bin/python -m pip install .
```
The pre-built package should be already installed in the parallax project.

When multiple prebuilt `_ext.cpython-<ver>-darwin.so` files are present in `lib/`,
Parallax automatically loads the one matching the current Python runtime.

### Usage Example
```python
import mlx.core as mx
from parallax_extensions.ops import paged_attention_v1
```
