import warnings
from collections.abc import Iterable
from typing import Callable, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import FastICA


