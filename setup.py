from pathlib import Path

from setuptools import find_packages, setup

# import numpy as np
# from version import __version__


with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(
        name="dynamo-release",
        version="v1.0.9",
        python_requires=">=3.7",
        install_requires=[l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()],  # 'yt>=3.5.1',
        extras_require={
            "spatial": ["pysal>2.0.0"],
            "interactive_plots": ["plotly"],
            "network": ["networkx", "nxviz", "hiveplotlib"],
            "dimension_reduction": ["fitsne>=1.0.1", "dbmap>=1.1.2"],
            "test": ["sympy>=1.4", "networkx"],
            "bigdata_visualization": [
                "datashader>=0.9.0",
                "bokeh>=1.4.0",
                "holoviews>=1.9.2",
            ],
        },
        packages=find_packages(),
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
