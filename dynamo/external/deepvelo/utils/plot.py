import warnings

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "deepvelo.utils.plot is deprecated. Please use deepvelo.plot.plot instead.",
        DeprecationWarning,
    )

from ..plot.plot import *
