from anndata import AnnData, read_h5ad
import numpy as np
from .utils_moments import estimation

def stratify(arr, strata):
    s = np.unique(strata)
    return [arr[strata==s[i]] for i in range(len(s))]

def strat_mom(arr, strata, fcn_mom):
    x = stratify(arr, strata)
    return np.array([fcn_mom(y) for y in x])

def calc_mom_all_genes(T, adata, fcn_mom):
    ng = len(adata.var['Gene'])
    nT = len(np.unique(T))
    Mn = np.zeros((ng, nT))
    Mo = np.zeros((ng, nT))
    Mt = np.zeros((ng, nT))
    Mr = np.zeros((ng, nT))
    for g in range(ng):
        L = np.array(adata[:, g].layers['new'], dtype=float)
        U = np.array(adata[:, g].layers['old'], dtype=float)
        rho = L/(L+U+0.01)
        Mn[g] = strat_mom(L, T, fcn_mom)
        Mo[g] = strat_mom(U, T, fcn_mom)
        Mt[g] = strat_mom(L + U, T, fcn_mom)
        Mr[g] = strat_mom(rho, T, fcn_mom)
    return Mn, Mo, Mt, Mr

class MomData(AnnData):
    def __init__(self, adata, time_key='Time', has_nan=False):
        #self.data = adata
        self.__dict__ = adata.__dict__
        # calculate first and second moments from data
        self.times = np.array(self.obs[time_key].values, dtype=float)
        self.uniq_times = np.unique(self.times)
        nT = self.get_n_times()
        ng = self.get_n_genes()
        self.M = np.zeros((ng, nT))      # first moments (data)
        self.V = np.zeros((ng, nT))      # second moments (data)
        for g in range(ng):
            L = np.array(self[:, g].layers['new'], dtype=float)        # consider using the `adata.obs_vector`, `adata.var_vector` methods or accessing the array directly.
            if has_nan:
                self.M[g] = strat_mom(L, self.times, np.nanmean)
                self.V[g] = strat_mom(L, self.times, np.nanvar)
            else:
                self.M[g] = strat_mom(L, self.times, np.mean)
                self.V[g] = strat_mom(L, self.times, np.var)

    def get_n_genes(self):
        return self.var.shape[0] #len(self.var['Gene'])

    def get_n_cell(self):
        return self.obs.shape[0] #len(self.obs['Cell'])

    def get_n_times(self):
        return len(self.uniq_times)

class Estimation:
    def __init__(self, adata, adata_u=None, time_key='Time', param_ranges=None, has_nan=False):
        # initialize simulator
        self.data = MomData(adata, time_key, has_nan)
        self.data_u = MomData(adata_u, time_key, has_nan) if adata_u is not None else None
        if param_ranges is None:
            param_ranges = {'a': [0, 10], 'b': [0, 10], 'alpha_a': [10, 1000], \
            'alpha_i': [0, 10], 'beta': [0, 10], 'gamma': [0, 10]}
        self.param_ranges = param_ranges
        self.n_params = len(param_ranges)

    def param_array2dict(self, parr):
        if parr.ndim == 1:
            return {'a': parr[0], 'b': parr[1], 'alpha_a': parr[2], \
                'alpha_i': parr[3], 'beta': parr[4], 'gamma': parr[5]}
        else:
            return {'a': parr[:, 0], 'b': parr[:, 1], 'alpha_a': parr[:, 2], \
                'alpha_i': parr[:, 3], 'beta': parr[:, 4], 'gamma': parr[:, 5]}

    def fit_gene(self, gene_no, n_p0=10):
        estm = estimation(list(self.param_ranges.values()))
        if self.data_u is None:
            m = self.data.M[gene_no, :].T
            v = self.data.V[gene_no, :].T
            x_data = np.vstack((m, v))
            popt, cost = estm.fit_lsq(self.data.uniq_times, x_data, p0=None, n_p0=n_p0, experiment_type='nosplice')
        else:
            mu = self.data_u.M[gene_no, :].T
            ms = self.data.M[gene_no, :].T
            vu = self.data_u.V[gene_no, :].T
            vs = self.data.V[gene_no, :].T
            x_data = np.vstack((mu, ms, vu, vs))
            popt, cost = estm.fit_lsq(self.data.uniq_times, x_data, p0=None, n_p0=n_p0, experiment_type=None)
        return popt, cost

    def fit(self, n_p0=10):
        ng = self.data.get_n_genes()
        params = np.zeros((ng, self.n_params))
        costs = np.zeros(ng)
        for i in range(ng):
            params[i], costs[i] = self.fit_gene(i, n_p0)
        return params, costs
        
