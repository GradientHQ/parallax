import os
import sys

from mlx import extension
from setuptools import setup


def _set_macos_build_defaults() -> None:
    if sys.platform != "darwin":
        return

    deployment_target = os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "14.0")
    cmake_args = os.environ.get("CMAKE_ARGS", "")
    if "CMAKE_OSX_DEPLOYMENT_TARGET" not in cmake_args:
        os.environ["CMAKE_ARGS"] = (
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target} {cmake_args}"
        ).strip()


if __name__ == "__main__":
    _set_macos_build_defaults()
    setup(
        name="parallax_extensions",
        version="0.0.1",
        description="Parallax Metal op extensions.",
        ext_modules=[extension.CMakeExtension("lib._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["parallax_extensions"],
        package_data={"lib": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.10",
    )
