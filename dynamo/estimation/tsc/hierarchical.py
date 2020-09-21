from tqdm import tqdm
import numpy as np
from scipy.sparse import issparse
from ...tools.utils import (
    find_extreme,
    elem_prod,
)
from ..csc.utils_velocity import fit_linreg

def fit_labeling_synthesis(new, total, t, intercept=False, perc_left=None, perc_right=None):
    T = np.unique(t)
    K = np.zeros(len(T))
    R2 = np.zeros(len(T))
    for i in range(len(T)):
        n = new[t==T[i]]
        r = total[t==T[i]]
        eind = find_extreme(n, r, perc_left=perc_left, perc_right=perc_right)
        K[i], _, R2[i], _ = fit_linreg(r[eind], n[eind], intercept=intercept)
    return K, R2


def compute_gamma_synthesis(K, T):
    gamma, _, r2, _ = fit_linreg(T, -np.log(1-K))
    return gamma, r2


def compute_velocity_synthesis(N, R, gamma, t):
    k = 1-np.exp(-np.einsum('i,j->ij', t, gamma))
    V = elem_prod(gamma, N)/k - elem_prod(gamma, R)
    return V


def lin_reg_gamma_synthesis(R, N, time, perc_right=100):
    n_var = R.shape[0]
    K, R2, gamma, r2 = np.zeros(n_var), np.zeros(n_var), np.zeros(n_var), np.zeros(n_var)

    for i, r, n in tqdm(zip(np.range(n_var), R, N), 'Estimate gamma via linear regression of t vs. -ln(1-K)'):
        r = r.A if issparse(r) else r
        n = n.A if issparse(n) else n

        K[i], R2[i] = fit_labeling_synthesis(n, r, time, perc_right=perc_right)
        gamma[i], r2[i] = compute_gamma_synthesis(K[i], time)

    return gamma, r2, K, R2
