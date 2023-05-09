import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.base import issparse
from sklearn.utils import sparsefuncs

from ..configuration import DKM, DynamoAdataKeyManager
from ..dynamo_logger import (
    main_debug,
    main_finish_progress,
    main_info,
    main_info_insert_adata,
    main_info_insert_adata_layer,
    main_info_insert_adata_obs,
    main_info_insert_adata_obsm,
    main_info_insert_adata_uns,
    main_log_time,
    main_warning,
)



