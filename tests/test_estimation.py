import dynamo as dyn
import numpy as np
import pandas as pd
import scipy.sparse as sp


def test_fit_linreg():
    from dynamo.estimation.csc.utils_velocity import fit_linreg, fit_linreg_robust
    from sklearn.datasets import make_regression

    X0, y0 = make_regression(n_samples=100, n_features=1, noise=0.5, random_state=0)
    X1, y1 = make_regression(n_samples=100, n_features=1, noise=0.5, random_state=2)
    X = np.vstack([X0.T, X1.T])
    y = np.vstack([y0, y1])

    k, b, r2, all_r2 = fit_linreg(X, y, intercept=True)
    k_r, b_r, r2_r, all_r2_r = fit_linreg_robust(X, y, intercept=True)

    assert np.allclose(k, k_r, rtol=1)
    assert np.allclose(b, b_r, rtol=1)
    assert np.allclose(r2, r2_r, rtol=1)
    assert np.allclose(all_r2, all_r2_r, rtol=1)


if __name__ == "__main__":
    # test_fit_linreg()
    pass