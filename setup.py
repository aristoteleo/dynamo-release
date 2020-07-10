from setuptools import setup, find_packages
import numpy as np
from version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dynamo",
    version=__version__,
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.18.1",
        "pandas>=0.25.1",
        "scipy>=1.0",
        "scikit-learn>=0.19.1",
        "cvxopt>=1.2.3",
        "anndata>=0.6.18",
        "loompy>=2.0.12",
        "matplotlib>=3.1.0",
        "trimap>=1.0.11",
        "setuptools",
        "numdifftools>=0.9.39",
        "umap-learn>=0.4.3",
        "hdbscan>=0.8.26",
        "PATSY>=0.5.1",
        "statsmodels>=0.9.0",
        "numba>=0.46.0",
        "seaborn>=0.9.0",
        "colorcet>=2.0.1",
        "datashader>=0.9.0",
        "bokeh>=1.4.0",
        "holoviews>=1.9.2",
        "tqdm>=4.31.1",
    ],  #,  'fitsne>=1.0.1''pysal>=2.0.0', 'yt>=3.5.1', 'sympy>=1.4',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    include_dirs=[np.get_include()],
    author="Xiaojie Qiu, Yan Zhang",
    author_email="xqiu.sc@gmail.com",
    description="Mapping Vector Field of Single Cells",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD",
    url="https://github.com/aristoteleo/dynamo-release",
    download_url=f"https://github.com/aristoteleo/dynamo-release",
    keywords=[
        "VectorField",
        "singlecell",
        "velocity",
        "scNT-seq",
        "sci-fate",
        "NASC-seq",
        "scSLAMseq",
        "potential",
    ],
)
