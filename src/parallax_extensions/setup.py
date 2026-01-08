from mlx import extension
from setuptools import setup

if __name__ == "__main__":
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
