# Installation

## Quick install

dynamo can be installed via `conda` or `pip`. We recommend installing into a virtual
environment to avoid conflicts with other packages.

```bash
conda install -c conda-forge dynamo-release
```

or

```bash
pip install dynamo-release
```

To install the newest version of dynamo, you can git clone our repo and then pip install:

```bash
git clone https://github.com/aristoteleo/dynamo-release.git
pip install dynamo-release/ --user
```

Don't know how to get started with virtual environments or `conda`/`pip`? Check out the
[prerequisites](#prerequisites) section.

## Prerequisites

### Virtual environment

A virtual environment can be created with either `conda` or `venv`. We recommend using `conda`. We
currently support Python 3.10 - 3.12.

For `conda`, we recommend using the [Miniforge](https://github.com/conda-forge/miniforge)
distribution, which is generally faster than the official distribution and comes with conda-forge
as the default channel (where dynamo is hosted).

```bash
conda create -n dynamo-env python=3.10  # any python 3.10 to 3.12
conda activate dynamo-env
```

For `venv`, we recommend using [uv](https://github.com/astral-sh/uv).

```bash
pip install -U uv
uv venv .dynamo-env
source .dynamo-env/bin/activate  # for macOS and Linux
.scvi-env\Scripts\activate  # for Windows
```


