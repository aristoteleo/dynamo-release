"""Mapping Vector Field of Single Cells
"""

from .get_version import get_dynamo_version, get_version

__version__ = get_version(__file__)
del get_version

# from .get_version import get_dynamo_version
#
# __version__ = get_dynamo_version()

from . import configuration, est, ext, mv, pd, pl, pp, sample_data, shiny, sim, tl, vf
from .data_io import *
from .dynamo_logger import (
    Logger,
    LoggerManager,
    main_critical,
    main_exception,
    main_info,
    main_tqdm,
    main_warning,
)
from .get_version import get_all_dependencies_version, session_info

# alias
config = configuration
