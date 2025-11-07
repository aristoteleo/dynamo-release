import warnings

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "deepvelo.utils.scatter is deprecated. Please use deepvelo.plot.scatter instead.",
        DeprecationWarning,
    )

from ..plot.scatter import *
