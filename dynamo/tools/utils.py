import numpy as np
from scipy.sparse import issparse
from .moments import strat_mom

def cal_12_mom(data, t):
    t_uniq = np.unique(t)
    m, v = np.zeros((data.shape[0], len(t_uniq))), np.zeros((data.shape[0], len(t_uniq)))
    for i in range(data.shape[0]):
        data_ = np.array(data[i].A.flatten(), dtype=float) if issparse(data) else np.array(data[i], dtype=float)  # consider using the `adata.obs_vector`, `adata.var_vector` methods or accessing the array directly.
        m[i], v[i] = strat_mom(data_, t, np.nanmean), strat_mom(data_, t, np.nanvar)

    return m, v, t_uniq
