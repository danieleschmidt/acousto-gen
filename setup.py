"""
Setup configuration for Acousto-Gen package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [req for req in requirements if req and not req.startswith("#")]

setup(
    name="acousto-gen",
    version="1.0.0",
    author="Daniel Schmidt",
    author_email="daniel@example.com",
    description="Generative acoustic holography toolkit for 3D pressure field manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/acousto-gen",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "cuda": [
            "cupy-cuda11x>=11.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.22.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "vtk>=9.0.0",
            "mayavi>=4.8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "acousto-gen=main:main",
            "acousto-gen-calibrate=calibration.calibrate:main",
            "acousto-gen-simulate=simulations.simulate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "acousto_gen": [
            "data/*.json",
            "data/*.npz",
            "configs/*.yaml",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/danieleschmidt/acousto-gen/issues",
        "Source": "https://github.com/danieleschmidt/acousto-gen",
        "Documentation": "https://acousto-gen.readthedocs.io",
    },
)