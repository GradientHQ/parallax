# Copyright © 2023-2024 Apple Inc.

from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        name="ops",
        version="0.0.0",
        description="Metal op extensions.",
        ext_modules=[extension.CMakeExtension("ops._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["ops"],
        package_data={"ops": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.10",
    )
