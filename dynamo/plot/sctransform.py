from typing import Optional

from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sctransform_plot_fit(
    adata: AnnData,
    xaxis: str = "gmean",
    fig: Optional[Figure] = None,
) -> Figure:
    """Plot the fitting of model parameters in sctransform.

    Args:
        adata: annotated data matrix after sctransform.
        xaxis: the gene expression metric is plotted on the x-axis.
        fig: Matplotlib figure object to use for the plot. If not provided, a new figure is created.

    Returns:
        The matplotlib figure object containing the plot.
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

def plot_residual_var(
    adata: AnnData,
    topngenes: int = 10,
    label_genes: bool = True,
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot the relationship between the mean and variance of gene expression across cells, highlighting the genes with
    the highest residual variance.

    Args:
        adata: annotated data matrix after sctransform.
        topngenes: the number of genes with the highest residual variance to highlight in the plot.
        label_genes: whether to label the highlighted genes in the plot. If `topngenes` is large, labeling genes may
            lead to plotting error because of the space limitation.
        ax: the axes on which to draw the plot. If None, a new figure and axes are created.

    Returns:
        The Figure object if `ax` is not given, else None.
    """
    def vars(a, axis=None):
        """Helper function to calculate variance of sparse matrix by equation: var = mean(a**2) - mean(a)**2"""
        a_squared = a.copy()
        a_squared.data **= 2
        return a_squared.mean(axis) - np.square(a.mean(axis))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = None

    gene_attr = pd.DataFrame(adata.var['log10_gmean_sct'])
    # gene_attr = gene_attr.loc[gene_names]
    gene_attr["var"] = vars(adata.X, axis=0).tolist()[0]
    gene_attr["mean"] = adata.X.mean(axis=0).tolist()[0]
    gene_attr_sorted = gene_attr.sort_values(
        "var", ascending=False
    ).reset_index()
    topn = gene_attr_sorted.iloc[:topngenes]
    gene_attr = gene_attr_sorted.iloc[topngenes:]
    ax.set_xscale("log")

    ax.scatter(
        gene_attr["mean"], gene_attr["var"], s=1.5, color="black"
    )
    ax.scatter(topn["mean"], topn["var"], s=1.5, color="deeppink")
    ax.axhline(1, linestyle="dashed", color="red")
    ax.set_xlabel("mean")
    ax.set_ylabel("var")
    if label_genes:
        texts = [
            plt.text(row["mean"], row["var"], row["index"])
            for index, row in topn.iterrows()
        ]
    fig.tight_layout()
    return fig