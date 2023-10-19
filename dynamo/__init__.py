"""Mapping Vector Field of Single Cells
"""

from .get_version import get_version, get_dynamo_version

__version__ = get_version(__file__)
del get_version

# from .get_version import get_dynamo_version
#
# __version__ = get_dynamo_version()

from . import pp
from . import est
from . import tl
from . import vf
from . import pd
from . import pl
from . import mv
from . import shiny
from . import sim
from .data_io import *
from . import sample_data
from . import configuration
from . import ext

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
