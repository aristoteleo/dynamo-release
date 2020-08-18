# code adapted from https://github.com/lmcinnes/umap/blob/7e051d8f3c4adca90ca81eb45f6a9d1372c076cf/umap/plot.py
from ..configuration import _themes

import numpy as np
import pandas as pd
import numba
from warnings import warn

import matplotlib.colors
import matplotlib.cm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

# import bokeh.plotting as bpl
# import bokeh.transform as btr
# from bokeh.plotting import output_notebook, output_file, show
#
# import holoviews as hv
# import holoviews.operation.datashader as hd
