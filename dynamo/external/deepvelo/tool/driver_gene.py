from typing import Optional

from anndata import AnnData
import numpy as np
from scipy.stats import pearsonr
from scvelo import logging as logg
from sklearn.linear_model import LassoCV


def driver_gene(
    adata: AnnData,
    method: str = "lasso",  # "corr", "lasso"
    random_seed: int = 0,
    expr_key: str = "Ms",
    do_log: bool = False,
    norm_along: str = "genes",  # "cells", "genes"
) -> Optional[LassoCV]:

    if method not in ["corr", "lasso"]:
        raise ValueError("method must be either corr or lasso")

    if "velocity_pseudotime" not in adata.obs.keys():
        raise ValueError(
            "velocity_pseudotime is not in adata.obs. "
            "Please run scv.tl.velocity_pseudotime(adata) first",
        )

    assert expr_key in adata.layers.keys(), f"{expr_key} is not in adata.layers"
    if do_log:
        if f"log_{expr_key}" in adata.layers.keys():
            logg.info(f"reuse existing 'log_{expr_key}' (adata.layers)")
        else:
            adata.layers[f"log_{expr_key}"] = np.log1p(adata.layers[expr_key])
            logg.info(f"added 'log_{expr_key}' (adata.layers)")
        expr_key = f"log_{expr_key}"

    if norm_along == "cells":
        norm_axis = 0
    elif norm_along == "genes":
        norm_axis = 1
    else:
        raise ValueError("norm_along must be either cells or genes")

    if method == "corr":
        time_corr = []
        time_corr_pval = []
        for gene in adata.var_names:
            tcorr, pval = pearsonr(
                adata.obs["velocity_pseudotime"],
                adata[:, gene].layers[expr_key].flatten(),
            )
            time_corr.append(tcorr)
            time_corr_pval.append(pval)

        adata.var["time_corr"] = time_corr
        adata.var["time_corr_abs"] = np.abs(time_corr)
        adata.var["time_corr_pval"] = time_corr_pval

        logg.info("added 'time_corr' (adata.var)")
        logg.info("added 'time_corr_abs' (adata.var)")
        logg.info("added 'time_corr_pval' (adata.var)")

    elif method == "lasso":
        exprs = adata.layers[expr_key]
        # normalize gene expression to standard Z-score, each row is a sample
        exprs = (exprs - exprs.mean(axis=norm_axis, keepdims=True)) / (
            exprs.std(axis=norm_axis, keepdims=True) + 1e-8
        )

        target_time = np.array(adata.obs["velocity_pseudotime"]).reshape(-1, 1)
        # make target zero mean
        target_time = target_time - target_time.mean()

        try:
            lasso = LassoCV(cv=5, random_state=random_seed)
            lasso.fit(exprs, target_time)
        except ValueError:
            # if Gram matrix error, https://github.com/scikit-learn/scikit-learn/pull/22059)
            lasso = LassoCV(cv=5, random_state=random_seed, precompute=False)
            lasso.fit(exprs, target_time)
        lasso_score = lasso.score(exprs, target_time)
        lasso_coef = lasso.coef_
        lasso_coef_abs = np.abs(lasso_coef)

        adata.var["lasso_coef"] = lasso_coef
        adata.var["lasso_coef_abs"] = lasso_coef_abs

        logg.info("added 'lasso_coef' (adata.var)")
        logg.info("added 'lasso_coef_abs' (adata.var)")

        return lasso
