# Copyright © 2023-2024 Apple Inc.

from mlx import extension
from setuptools import setup

if __name__ == "__main__":
    setup(
        name="parallax_ops",
        version="0.0.1",
        description="Metal op extensions.",
        ext_modules=[extension.CMakeExtension("parallax_ops._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["parallax_ops"],
        package_data={"parallax_ops": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.10",
    )
