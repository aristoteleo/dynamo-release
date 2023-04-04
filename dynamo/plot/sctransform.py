from typing import Optional, Union

from anndata import AnnData

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sctransform_plot_fit(adata, xaxis="gmean", fig=None):
    """
    Parameters
    ----------
    pysct_results: dict
                   obsect returned by pysctransform.vst
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 3))
    gene_names = adata.var['genes_step1_sct'][
        ~adata.var['genes_step1_sct'].isna()].index

    genes_log10_mean = adata.var["log10_gmean_sct"]
    genes_log_gmean = genes_log10_mean[~genes_log10_mean.isna()]

    model_params_fit = pd.concat(
        [adata.var["log_umi_sct"], adata.var["Intercept_sct"], adata.var["theta_sct"]], axis=1)
    model_params = pd.concat(
        [adata.var["log_umi_step1_sct"], adata.var["Intercept_step1_sct"], adata.var["model_pars_theta_step1"]],
        axis=1)
    model_params_fit = model_params_fit.rename(
        columns={"log_umi_sct": "log_umi", "Intercept_sct": "Intercept", "theta_sct": "theta"})
    model_params = model_params.rename(
        columns={"log_umi_step1_sct": "log_umi",
                 "Intercept_step1_sct": "Intercept",
                 "model_pars_theta_step1": "theta"})

    model_params = model_params.loc[gene_names]

    total_params = model_params_fit.shape[1]

    for index, column in enumerate(model_params_fit.columns):
        ax = fig.add_subplot(1, total_params, index + 1)
        model_param_col = model_params[column]

        # model_param_outliers = is_outlier(model_param_col)
        if column != "theta":
            ax.scatter(
                genes_log_gmean,  # [~model_param_outliers],
                model_param_col,  # [~model_param_outliers],
                s=1,
                label="single gene estimate",
                color="#2b8cbe",
            )
            ax.scatter(
                genes_log10_mean,
                model_params_fit[column],
                s=2,
                label="regularized",
                color="#de2d26",
            )
            ax.set_ylabel(column)
        else:
            ax.scatter(
                genes_log_gmean,  # [~model_param_outliers],
                np.log10(model_param_col),  # [~model_param_outliers],
                s=1,
                label="single gene estimate",
                color="#2b8cbe",
            )
            ax.scatter(
                genes_log10_mean,
                np.log10(model_params_fit[column]),
                s=2,
                label="regularized",
                color="#de2d26",
            )
            ax.set_ylabel("log10(" + column + ")")
        if column == "od_factor":
            ax.set_ylabel("log10(od_factor)")

        ax.set_xlabel("log10(gene_{})".format(xaxis))
        ax.set_title(column)
        ax.legend(frameon=False)
    _ = fig.tight_layout()
    return fig

def plot_residual_var(adata, topngenes=30, label_genes=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = None

    gene_names = adata.var['genes_step1_sct'][~adata.var['genes_step1_sct'].isna()].index
    genes_log10_mean = adata.var["log10_gmean_sct"][~adata.var["log10_gmean_sct"].isna()]
    gene_attr = pd.DataFrame(adata.var["dispersion_step1_sct"])
    gene_attr = gene_attr.loc[gene_names]
    gene_attr["genes_log10_step1_mean"] = genes_log10_mean

    gene_attr_sorted = gene_attr.sort_values(
        "dispersion_step1_sct", ascending=False
    ).reset_index()
    # TODO: error check
    topn = gene_attr_sorted.iloc[:topngenes]
    gene_attr = gene_attr_sorted.iloc[topngenes:]
    ax.set_xscale("log")

    ax.scatter(
        gene_attr["genes_log10_step1_mean"], gene_attr["dispersion_step1_sct"], s=1.5, color="black"
    )
    ax.scatter(topn["genes_log10_step1_mean"], topn["dispersion_step1_sct"], s=1.5, color="deeppink")
    ax.axhline(1, linestyle="dashed", color="red")
    ax.set_xlabel("genes_log10_step1_mean")
    ax.set_ylabel("dispersion_step1_sct")
    # if label_genes:
    #     from adjustText import adjust_text
    #     texts = [
    #         plt.text(row["genes_log10_step1_mean"], row["dispersion_step1_sct"], row["index"])
    #         for index, row in topn.iterrows()
    #     ]
    #     adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5))
    fig.tight_layout()
    return fig