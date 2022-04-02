# code adapted from https://github.com/lmcinnes/umap/blob/7e051d8f3c4adca90ca81eb45f6a9d1372c076cf/umap/plot.py
from warnings import warn

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..configuration import _themes

# import bokeh.plotting as bpl
# import bokeh.transform as btr
# from bokeh.plotting import output_notebook, output_file, show
#
# import holoviews as hv
# import holoviews.operation.datashader as hd
