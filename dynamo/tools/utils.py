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

def get_U_S_for_velocity_estimation(subset_adata, has_splicing, log_unnormalized):
    if has_splicing:
        if 'X_uu' in subset_adata.layers.keys():  # unlabel spliced: S
            uu, ul, su, sl = subset_adata.layers['X_uu'].T, subset_adata.layers['X_ul'].T, \
                             subset_adata.layers['X_su'].T, subset_adata.layers['X_sl'].T
        else:
            uu, ul, su, sl = subset_adata.layers['uu'].T, subset_adata.layers['ul'].T, \
                             subset_adata.layers['su'].T, subset_adata.layers['sl'].T
            if issparse(uu):
                uu.data = np.log(uu.data + 1) if log_unnormalized else uu.data
                ul.data = np.log(ul.data + 1) if log_unnormalized else ul.data
                su.data = np.log(su.data + 1) if log_unnormalized else su.data
                sl.data = np.log(sl.data + 1) if log_unnormalized else sl.data
            else:
                uu = np.log(uu + 1) if log_unnormalized else uu
                ul = np.log(ul + 1) if log_unnormalized else ul
                su = np.log(su + 1) if log_unnormalized else su
                sl = np.log(sl + 1) if log_unnormalized else sl

        U, S = ul, sl
    else:
        if 'X_new' in subset_adata.layers.keys():  # run new / total ratio (NTR)
            U = subset_adata.layers['X_new'].T
            S = subset_adata.layers['X_total'].T - subset_adata.layers['X_new'].T
        elif 'new' in subset_adata.layers.keys():
            U = subset_adata.layers['new'].T
            S = subset_adata.layers['total'].T - subset_adata.layers['new'].T
            if issparse(U):
                U.data = np.log(U.data + 1) if log_unnormalized else U.data
                S.data = np.log(S.data + 1) if log_unnormalized else S.data
            else:
                U = np.log(U + 1) if log_unnormalized else U
                S = np.log(S + 1) if log_unnormalized else S

    return U, S
