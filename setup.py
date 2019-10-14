from setuptools import setup, find_packages
import numpy as np
from version import __version__ 

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dynamo",
    version=__version__,
    install_requires=['numpy>=1.14', 'scipy>=1.0', 'scikit-learn>=0.19.1', "cvxopt>=1.2.3",
                      'anndata>=0.6.18', 'loompy>=2.0.12', 'matplotlib>=2.2', "trimap>=1.0.11",
                      'setuptools', 'seaborn>=0.9.0', 'sympy>=1.4', 'numdifftools>=0.9.39',
                      'yt>=3.5.1', 'plotnine>=0.5.1', 'umap-learn>=0.3.9', 'pysal>=2.0.0',
                      'statsmodels>=0.9.0'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    include_dirs=[np.get_include()],
    author="Xiaojie Qiu, Yan Zhang",
    author_email="xqiu.sc@gmail.com",
    description='Mapping Vector Field of Single Cells',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='BSD',
    url="https://github.com/aristoteleo/dynamo-release",
    download_url=f"https://github.com/aristoteleo/dynamo-release",
    keywords=["VectorField", "singlecell", "velocity", "scSLAMseq", "potential"]
    )
