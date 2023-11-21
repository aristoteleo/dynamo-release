"""Mapping Vector Field of Single Cells
"""

# module to deal with reaction/diffusion/advection.
# code was loosely based on PBA, WOT and PRESCIENT.

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors

from .connectivity import k_nearest_neighbors

def score_cells(
    adata: AnnData,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    basis: Optional[str] = None,
    n_neighbors: int = 30,
    beta: float = 0.1,
    iteration: int = 5,
    metric: Union[str, Callable] = "euclidean",
    metric_kwds: Optional[Dict[str, Any]] = None,
    cores: int = 1,
    seed: int = 19491001,
    return_score: bool = True,
    **kwargs,
) -> Optional[np.ndarray]:
    """Score cells based on a set of genes.

    Args:
        adata: An AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        genes: The gene names whose gene expression will be used for predicting cell fate. By default (when genes is set
            to None), the genes used for velocity embedding (var.use_for_transition) will be used for vector field
            reconstruction. Note that the genes to be used need to have velocity calculated and corresponds to those
            used in the `dyn.tl.VectorField` function. Defaults to None.
        layer: Which layer of the data will be used for predicting cell fate with the reconstructed vector field
            function. The layer once provided, will override the `basis` argument and then predicting cell fate in high
            dimensional space. Defaults to None.
        basis: The embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the
            reconstructed trajectory will be projected back to high dimensional space via the `inverse_transform`
            function. Defaults to None.
        n_neighbors: Number of nearest neighbors. Defaults to 30.
        beta: The weight that will apply to the current query cell. Defaults to 0.1.
        iteration: Number of smooth iterations. Defaults to 5.
        metric: The distance metric to use for the tree.  The default metric is , and with p=2 is equivalent to the
            standard Euclidean metric. See the documentation of :class:`DistanceMetric` for a list of available metrics.
            If metric is "precomputed", X is assumed to be a distance matrix and must be square during fit. X may be a
            `sparse graph`, in which case only "nonzero" elements may be considered neighbors. Defaults to "euclidean".
        metric_kwds: Additional keyword arguments for the metric function. Defaults to None.
        cores: The number of parallel jobs to run for neighbors search. `None` means 1 unless in a
            `joblib.parallel_backend` context. `-1` means using all processors. Defaults to 1.
        seed: Random seed to ensure the reproducibility of each run. Defaults to 19491001.
        return_score: Whether to return the score. If False, save the smoothed score to `cell_scores` column in the
            `.obs` attribute and also to the dictionary corresponding to the `score_cells` key in the .uns attribute.
            Defaults to True.

    Raises:
        ValueError: `X_pca` unavailable in `.obsm`.
        ValueError: `basis` not available in `.obsm`.
        ValueError: `genes` not provided and no "use_for_pca" in .obs.
        ValueError: Input genes have no overlap with genes in the AnnData object.

    Returns:
        The calculated cell scores if `return` score is true, otherwise the scores would be updated as annotations of
        the AnnData object inplace.
    """

    if basis is None and "X_pca" not in adata.obsm.keys():
        raise ValueError(f"Your adata doesn't have 'X_pca' basis in .obsm.")
    elif basis is not None and "X_" + basis not in adata.obsm.keys():
        raise ValueError(f"Your adata doesn't have the {basis} you inputted in .obsm attribute of your adata.")

    if genes is None and "use_for_pca" not in adata.var.keys():
        raise ValueError(f"Your adata doesn't have 'use_for_pca' column in .var.")

    if genes is None:
        genes = adata.var_names[adata.var.use_for_pca]
    else:
        genes = (
            list(adata.var_names.intersection(genes))
            if adata.var_names[0].isupper()
            else list(adata.var_names.intersection([i.capitalize() for i in genes]))
            if adata.var_names[0][0].isupper() and adata.var_names[0][1:].islower()
            else list(adata.var_names.intersection([i.lower() for i in genes]))
        )

    if len(genes) < 1:
        raise ValueError(f"Your inputted gene list doesn't overlap any gene in your adata object.")

    X_basis = adata.obsm["X_pca"] if basis is None else adata.obsm["X_" + basis]

    knn, distances = k_nearest_neighbors(
        X_basis,
        k=n_neighbors - 1,
        metric=metric,
        metric_kwads=metric_kwds,
        exclude_self=False,
        pynn_rand_state=seed,
        n_jobs=cores,
        **kwargs,
    )

    X_data = adata[:, genes].X if layer in [None, "X"] else adata[:, genes].layers[layer]

    prev_score = X_data.mean(1).A1 if issparse(X_data) else X_data.mean(1)
    cur_score = np.zeros(prev_score.shape)

    for _ in range(iteration):
        for i in range(len(prev_score)):
            xn = prev_score[knn[i]]
            cur_score[i] = (beta * xn[0]) + ((1 - beta) * xn[1:].mean(axis=0))
        prev_score = cur_score

    smoothed_score = cur_score

    if return_score:
        return smoothed_score
    else:
        adata.uns["score_cells"] = {
            "smoothed_score": smoothed_score,
            "genes": genes,
            "layer": layer,
            "basis": basis,
        }
        adata.obs["cell_score"] = smoothed_score


def cell_growth_rate(
    adata: AnnData,
    group: str,
    source: Optional[str] = None,
    target: Optional[str] = None,
    L0: float = 0.3,
    L: float = 1.2,
    k: float = 1e-3,
    birth_genes: Optional[List[str]] = None,
    death_genes: Optional[List[str]] = None,
    clone_column: Optional[str] = None,
    **kwargs,
) -> AnnData:
    """Estimate the growth rate via clone information or logistic equation of population dynamics.

    Args:
        adata: An AnnData object.
        group: The column key in .obs points to the collection time of each cell, required for calculating growth rate
            with clone information.
        source: The value in the `group` column representing the starting point from collection time of each cell,
            required for calculating growth rate with clone information.
        target: The value in the `group` column representing the end point from collection time of each cell, required
            for calculating growth rate with clone information.
        L0: The base growth/death rate. Defaults to 0.3.
        L: The maximum growth/death rate. Defaults to 1.2.
        k: The steepness of the curve. Defaults to 1e-3.
        birth_genes: The gene list associated with the cell cycle process. If None, GSEA's KEGG_CELL_CYCLE will be used.
            Defaults to None.
        death_genes: The gene list associated with the cell cycle process. If None, GSEA's KEGG_APOPTOSIS will be used.
            Defaults to None.
        clone_column: The column key in .obs points to the clone id if there is any. If a cell doesn't belong to any
            clone, the clone id of that cell should be assigned as `np.nan`. Defaults to None.
        kwargs: Additional arguments that will be passed to score_cells function.

    Raises:
        ValueError: `clone_name` or `group` not in .obs.
        ValueError: `source` or `target` not in .obs.

    Returns:
        An updated adata object that includes `growth_rate` column or `growth_rate, birth_score, death_score` in its
        `.obs` attribute when the clone based or purely expression based growth rate was calculated.
    """

    # calculate growth rate when there is clone information.
    all_clone_info = [clone_column, group, source, target]

    obs = adata.obs
    source_mask_, target_mask_ = (
        obs[group].values == source,
        obs[group].values == target,
    )

    if all(i is not None for i in all_clone_info):

        if any(i not in adata.obs.keys() for i in all_clone_info[:2]):
            raise ValueError(
                f"At least one of your input clone information {clone_column}, {group} "
                f"is not in your adata .obs attribute."
            )
        if any(i not in adata.obs[group].values for i in all_clone_info[2:]):
            raise ValueError(
                f"At least one of your input source/target information {source}, {target} "
                f"is not in your adata.obs[{group}] column."
            )

        clone_time_count = obs.groupby([clone_column])[group].value_counts().unstack().fillna(0).astype(int)
        source_meta = obs.loc[source_mask_]
        source_mask = (source_meta[clone_column] != np.nan).values

        target_meta = obs.loc[target_mask_]
        target_mask = (target_meta[clone_column] != np.nan).values

        source_num = clone_time_count.loc[source_meta.loc[source_mask, clone_column], source].values + 1
        target_num = clone_time_count.loc[target_meta.loc[target_mask, clone_column], target].values + 1

        growth_rates = target_num / source_num
    else:
        # calculate growth rate when there is no clone information.
        if birth_genes is None:
            birth_genes = pd.read_csv(
                "https://raw.githubusercontent.com/Xiaojieqiu/jungle/master/Cell_cycle.txt",
                header=None,
                dtype=str,
            )
            birth_genes = birth_genes[0].values

        if death_genes is None:
            death_genes = pd.read_csv(
                "https://raw.githubusercontent.com/Xiaojieqiu/jungle/master/Apoptosis.txt",
                header=None,
                dtype=str,
            )
            death_genes = death_genes[0].values

        birth_score = score_cells(adata, genes=birth_genes, **kwargs)
        death_score = score_cells(adata, genes=death_genes, **kwargs)
        adata.obs["birth_score"] = birth_score
        adata.obs["death_score"] = death_score

        kb = np.log(k) / np.min(birth_score)
        kd = np.log(k) / np.min(death_score)

        b = birth_score[source_mask_]
        d = death_score[source_mask_]

        b = L0 + L / (1 + np.exp(-kb * b))
        d = L0 + L / (1 + np.exp(-kd * d))
        growth_rates = b - d

    adata.obs["growth_rate"] = np.nan
    adata.obs.loc[source_mask_, "growth_rate"] = growth_rates

    return adata


def n_descentants(birth: npt.ArrayLike, death: npt.ArrayLike, dt: npt.ArrayLike) -> npt.ArrayLike:
    """Calculate the number of descentants for a given cell after some given time.

    Args:
        birth: Logged birth number per unit time.
        death: Logged death number per unit time.
        dt: Total time.

    Returns:
        The number of descentants for a given cell after some given time.
    """
    return np.exp(dt * (birth - death))


def growth_rate(n: npt.ArrayLike, dt: npt.ArrayLike) -> npt.ArrayLike:
    """Calculate the logged growth rate.

    Args:
        n: Increased cell number.
        dt: Time interval.

    Returns:
        The growth rate.
    """
    return np.log(n) / dt
