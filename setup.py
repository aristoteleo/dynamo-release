from pathlib import Path

from setuptools import find_packages, setup
import os
# from version import __version__


def read_requirements(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if not line.isspace()]


with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(
        name="dynamo-release",
        version="v1.4.0",
        python_requires=">=3.7",
        install_requires=read_requirements("requirements.txt"),
        extras_require={
            "docs": read_requirements(os.path.join("docs", "requirements.txt")),
        },
        packages=find_packages(exclude=("tests", "docs")),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
        ],
        #     include_dirs=[np.get_include()],
        author="Xiaojie Qiu, Yan Zhang, Ke Ni",
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
