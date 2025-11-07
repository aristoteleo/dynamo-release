from typing import Optional, Tuple

import numpy as np
import scvelo as scv
from scvelo import logging as logg
from scvelo.core import sum as sum_
from anndata import AnnData

from ..utils.plot import dist_plot


def clip_and_norm_Ms_Mu(
    adata,
    do_clip: bool = True,
    do_norm: bool = True,
    target_mean: float = 0.4,
    replace: bool = False,
    save_fig: Optional[str] = None,
    plot: bool = True,
    print_summary: bool = True,
) -> Tuple[float, float]:
    """
    Normalize using the mean and standard deviation of the gene expression matrix.

    Args:
        adata (Anndata): Anndata object.
        target_mean (float): target mean.
        replace (bool): replace the original data.
        save_fig (str): directory to save figures.
        plot (bool): plot the distribution of the normalized data.
        print_summary (bool): print the summary of the normalized data.

    Returns:
        Tupel[float, float]: scale factor for Ms and Mu.
    """
    non_zero_Ms = adata.layers["Ms"][adata.layers["Ms"] > 0]
    non_zero_Mu = adata.layers["Mu"][adata.layers["Mu"] > 0]
    if print_summary:
        print(
            f"Raw Ms: mean {adata.layers['Ms'].mean():.2f},"
            f" max {adata.layers['Ms'].max():.2f},"
            f" std {adata.layers['Ms'].std():.2f},"
            f" 99.5% quantile {np.percentile(adata.layers['Ms'], 99.5):.2f}"
            f" 99.5% of non-zero: {np.percentile(non_zero_Ms, 99.5):.2f}"
        )
        print(
            f"Raw Mu: mean {adata.layers['Mu'].mean():.2f},"
            f" max {adata.layers['Mu'].max():.2f},"
            f" std {adata.layers['Mu'].std():.2f},"
            f" 99.5% quantile {np.percentile(adata.layers['Mu'], 99.5):.2f}"
            f" 99.5% of non-zero: {np.percentile(non_zero_Mu, 99.5):.2f}"
        )

    if do_clip:
        # clip the max value to 99.5% quantile
        adata.layers["NMs"] = np.clip(
            adata.layers["Ms"], None, np.percentile(non_zero_Ms, 99.5)
        )
        adata.layers["NMu"] = np.clip(
            adata.layers["Mu"], None, np.percentile(non_zero_Mu, 99.5)
        )
    else:
        adata.layers["NMs"] = adata.layers["Ms"]
        adata.layers["NMu"] = adata.layers["Mu"]
    logg.hint(f"added 'NMs' (adata.layers)")
    logg.hint(f"added 'NMu' (adata.layers)")

    if plot:
        dist_plot(
            adata.layers["NMs"].flatten(),
            adata.layers["NMu"].flatten(),
            bins=20,
            labels=["NMs", "NMu"],
            title="Distribution of Ms and Mu",
            save=f"{save_fig}/hist-Ms-Mu.png" if save_fig is not None else None,
        )

    scale_Ms, scale_Mu = 1.0, 1.0
    if do_norm:
        scale_Ms = adata.layers["NMs"].mean() / target_mean
        scale_Mu = adata.layers["NMu"].mean() / target_mean
        adata.layers["NMs"] = adata.layers["NMs"] / scale_Ms
        adata.layers["NMu"] = adata.layers["NMu"] / scale_Mu
        print(f"Normalized Ms and Mu to mean of {target_mean}")
        if plot:
            ax = scv.pl.hist(
                [adata.layers["NMs"].flatten(), adata.layers["NMu"].flatten()],
                labels=["NMs", "NMu"],
                kde=False,
                normed=False,
                bins=20,
                # xlim=[0, 1],
                fontsize=18,
                legend_fontsize=16,
                show=False,
            )
            if save_fig is not None:
                ax.get_figure().savefig(f"{save_fig}/hist-normed-Ms-Mu.png")

    if print_summary:
        print(
            f"New Ms: mean {adata.layers['NMs'].mean():.2f},"
            f" max {adata.layers['NMs'].max():.2f},"
            f" std {adata.layers['NMs'].std():.2f},"
            f" 99.5% quantile {np.percentile(adata.layers['NMs'], 99.5):.2f}"
        )
        print(
            f"New Mu: mean {adata.layers['NMu'].mean():.2f},"
            f" max {adata.layers['NMu'].max():.2f},"
            f" std {adata.layers['NMu'].std():.2f},"
            f" 99.5% quantile {np.percentile(adata.layers['NMu'], 99.5):.2f}"
        )

    if replace:
        adata.layers["Ms"] = adata.layers["NMs"]
        adata.layers["Mu"] = adata.layers["NMu"]
        logg.hint(f"replaced 'Ms' (adata.layers) with 'NMs'")
        logg.hint(f"replaced 'Mu' (adata.layers) with 'NMu'")

    return scale_Ms, scale_Mu


def autoset_coeff_s(adata: AnnData, use_raw: bool = True) -> float:
    """
    Automatically set the weighting for objective term of the spliced
    read correlation. Modified from the scv.pl.proportions function.

    Args:
        adata (Anndata): Anndata object.
        use_raw (bool): use raw data or processed data.

    Returns:
        float: weighting coefficient for objective term of the unpliced read
    """
    layers = ["spliced", "unspliced", "ambigious"]
    layers_keys = [key for key in layers if key in adata.layers.keys()]
    counts_layers = [sum_(adata.layers[key], axis=1) for key in layers_keys]

    if use_raw:
        ikey, obs = "initial_size_", adata.obs
        counts_layers = [
            obs[ikey + layer_key] if ikey + layer_key in obs.keys() else c
            for layer_key, c in zip(layers_keys, counts_layers)
        ]
    counts_total = np.sum(counts_layers, 0)
    counts_total += counts_total == 0
    counts_layers = np.array([counts / counts_total for counts in counts_layers])
    counts_layers = np.mean(counts_layers, axis=1)

    spliced_counts = counts_layers[layers_keys.index("spliced")]
    ratio = spliced_counts / counts_layers.sum()

    if ratio < 0.7:
        coeff_s = 0.5
        print(
            f"The ratio of spliced reads is {ratio*100:.1f}% (less than 70%). "
            f"Suggest using coeff_s {coeff_s}."
        )
    elif ratio < 0.85:
        coeff_s = 0.75
        print(
            f"The ratio of spliced reads is {ratio*100:.1f}% (between 70% and 85%). "
            f"Suggest using coeff_s {coeff_s}."
        )
    else:
        coeff_s = 1.0
        print(
            f"The ratio of spliced reads is {ratio*100:.1f}% (more than 85%). "
            f"Suggest using coeff_s {coeff_s}."
        )

    return coeff_s
