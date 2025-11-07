from typing import List, Optional, Tuple
import warnings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from anndata import AnnData
from scvelo import logging as logg
from scvelo.core import l2_norm, prod_sum
from scvelo.preprocessing.neighbors import get_neighs

from .util import get_indices

# from ..trainer import Trainer

# Code modified from function velocity_confidence from
# https://github.com/theislab/scvelo/blob/master/scvelo/tools/velocity_confidence.py
def velocity_confidence(
    data,
    vkey="velocity",
    method="corr",
    scope_key=None,
    copy=False,
    only_velocity_genes=False,
    only_high_spearman=False,
):
    """Computes confidences of velocities.

    .. code:: python

        scv.tl.velocity_confidence(adata)
        scv.pl.scatter(adata, color='velocity_confidence', perc=[2,98])

    .. image:: https://user-images.githubusercontent.com/31883718/69626334-b6df5200-1048-11ea-9171-495845c5bc7a.png
       :width: 600px


    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    method: `str` (default: `'corr'`), choice of `'corr'` or `'cosine'`
        Method to use for computing confidence, whether to use correlation or
        cosine similarity.
    scope_key: `str` (default: `None`)
        For each cell, cells with in scope_key are used for computing confidence.
        If `None`, use cell neighbors. Else, pick the cells in scope_key. Valid
        scope_key has to be in adata.obs.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to adata.
    only_velocity_genes: `bool` (default: `False`)
        Only use velocity genes.
    only_high_spearman: `bool` (default: `False`)
        Only use high spearman.

    Returns
    -------
    velocity_length: `.obs`
        Length of the velocity vectors for each individual cell
    velocity_confidence: `.obs`
        Confidence for each cell
    """  # noqa E501

    adata = data.copy() if copy else data
    if vkey not in adata.layers.keys():
        raise ValueError("You need to run `tl.velocity` first.")
    if method not in ["corr", "cosine"]:
        raise ValueError("Method must be either 'corr' or 'cosine'.")
    if scope_key is not None and method == "corr":
        raise ValueError("Cannot use scope_key with method 'corr'.")

    V = np.array(adata.layers[vkey])

    # filter genes if needed
    tmp_filter = np.invert(np.isnan(np.sum(V, axis=0)))
    if only_velocity_genes and (f"{vkey}_genes" in adata.var.keys()):
        tmp_filter &= np.array(adata.var[f"{vkey}_genes"], dtype=bool)
    if only_high_spearman and ("spearmans_score" in adata.var.keys()):
        tmp_filter &= adata.var["spearmans_score"].values > 0.1
    V = V[:, tmp_filter]

    # zero mean, only need for correlation
    if method == "corr":
        V -= V.mean(1)[:, None]
    V_norm = l2_norm(V, axis=1)
    # normalize, only need for cosine similarity
    if method == "cosine":
        V /= V_norm[:, None]
    R = np.zeros(adata.n_obs)

    indices = (
        get_indices(dist=get_neighs(adata, "distances"))[0]
        if not scope_key
        else adata.obs[scope_key]  # the scope_key (e.g. cluster) of each cell
    )
    Vi_neighs_avg_cache = {}
    for i in range(adata.n_obs):
        if not scope_key:
            # use the neighbors of the cell
            Vi_neighs = V[indices[i]]
        else:
            # use the cells in scope_key
            if indices[i] not in Vi_neighs_avg_cache:
                Vi_neighs = V[indices == indices[i]]
                Vi_neighs_avg_cache[indices[i]] = Vi_neighs.mean(0)
        if method == "corr":
            Vi_neighs -= Vi_neighs.mean(1)[:, None]
            R[i] = np.mean(
                np.einsum("ij, j", Vi_neighs, V[i])
                / (l2_norm(Vi_neighs, axis=1) * V_norm[i])[None, :]
            )
        elif method == "cosine":
            # could compute mean first, because V has been normed
            Vi_neighs_avg = (
                Vi_neighs_avg_cache[indices[i]] if scope_key else Vi_neighs.mean(0)
            )
            R[i] = np.inner(V[i], Vi_neighs_avg)

    adata.obs[f"{vkey}_length"] = V_norm.round(2)
    adata.obs[f"{vkey}_confidence_{method}"] = R

    logg.hint(f"added '{vkey}_length' (adata.obs)")
    logg.hint(f"added '{vkey}_confidence_{method}' (adata.obs)")

    # if f"{vkey}_confidence_transition" not in adata.obs.keys():
    #     velocity_confidence_transition(adata, vkey)

    return adata if copy else None


def split_by_cluster():
    """split scores by cluster."""
    pass


# Code modified from function inner_cluster_coh from
# https://github.com/qiaochen/VeloAE/blob/main/veloproj/eval_util.py
def velocity_cosine(
    data,
    vkey="velocity",
    copy=False,
):
    pass


def inner_cluster_coh(adata, k_cluster, k_velocity, return_raw=False):
    """In-cluster Coherence Score.

    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        return_raw (bool): return aggregated or raw scores.

    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.

    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}
    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns["neighbors"]["indices"][sel]
        same_cat_nodes = map(lambda nodes: keep_type(adata, nodes, cat, k_cluster), nbs)
        velocities = adata.layers[k_velocity]
        cat_vels = velocities[sel]
        cat_score = [
            cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean()
            for ith, nodes in enumerate(same_cat_nodes)
            if len(nodes) > 0
        ]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])


# Code modified from cross_boundary_correctness
# https://github.com/qiaochen/VeloAE/blob/main/veloproj/eval_util.py
def cross_boundary_correctness(
    adata: AnnData,
    k_cluster: str,
    k_velocity: str,
    cluster_edges: List[Tuple[str, str]],
    return_raw: bool = False,
    x_emb_key: str = "umap",
    inplace: bool = True,
    output_key_prefix: str = "",
):
    """Cross-Boundary Direction Correctness Score (A->B)

    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        return_raw (bool): return aggregated or raw scores.
        x_emb (str): key to x embedding. If one of the keys in adata.layers, then
            will use the layer and compute the score in the raw space. Otherwise,
            will use the embedding in adata.obsm. Default to "umap".
        inplace (bool): whether to add the score to adata.obs.
        output_key_prefix (str): prefix to the output key. Defaults to "".

    Returns:
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        and
        dict: all_scores indexed by cluster_edges if return_raw is True.

    """

    def _select_emb(adata: AnnData, k_velocity: str, x_emb_key: str):
        if x_emb_key in adata.layers.keys():
            # using embedding from raw space
            x_emb = adata.layers[x_emb_key]
            v_emb = adata.layers[k_velocity]

        else:  # embedding from visualization dimensions
            if x_emb_key.startswith("X_"):
                v_emb_key = k_velocity + x_emb_key[1:]
            else:
                v_emb_key = k_velocity + "_" + x_emb_key
                x_emb_key = "X_" + x_emb_key
            assert x_emb_key in adata.obsm.keys()
            assert v_emb_key in adata.obsm.keys()
            x_emb = adata.obsm[x_emb_key]
            v_emb = adata.obsm[v_emb_key]
        return x_emb, v_emb

    scores = {}
    all_scores = {}
    x_emb, v_emb = _select_emb(adata, k_velocity, x_emb_key)

    for u, v in cluster_edges:
        assert u in adata.obs[k_cluster].cat.categories, f"cluster {u} not found"
        assert v in adata.obs[k_cluster].cat.categories, f"cluster {v} not found"
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns["neighbors"]["indices"][sel]  # [n * 30]

        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.mean(dir_scores))

        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score

    all_cbcs_ = np.concatenate([np.array(d) for d in all_scores.values()])

    if inplace:
        adata.uns[f"{output_key_prefix}direction_scores"] = scores
        adata.uns[f"{output_key_prefix}raw_direction_scores"] = all_cbcs_
        logg.info(f"added '{output_key_prefix}direction_scores' (adata.uns)")
        logg.info(f"added '{output_key_prefix}raw_direction_scores' (adata.uns)")

    if return_raw:
        return scores, np.mean(all_cbcs_), all_scores

    return scores, np.mean(all_cbcs_)


def genewise_cross_boundary_correctness(
    adata: AnnData,
    cluster_key: str,
    velocity_key: str,
    cluster_edges: List[Tuple[str, str]],
    spliced_key: str = "Ms",
    unspliced_key: str = "Mu",
    inplace: bool = True,
):
    """
    computes Cross-Boundary Direction Correctness Score (A->B) per gene on the
    spliced unspliced phase dimensions. Output scores will be written to adata.var.

    Args:
        adata (Anndata): Anndata object.
        cluster_key (str): key to the cluster column in adata.obs DataFrame.
        velocity_key (str): key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        spliced_key (str): key to the spliced phase in adata.layers.
        unspliced_key (str): key to the unspliced phase in adata.layers.
        return_raw (bool): return aggregated or raw scores.
        inplace (bool): whether to add the score to adata.obs.

    Returns:
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        and
        dict: all_scores indexed by cluster_edges if return_raw is True.

    """

    gene_wise_scores_on_boundaries = {}
    x_emb = np.stack(
        [adata.layers[spliced_key], adata.layers[unspliced_key]],
        axis=2,
    )  # (n_cells, n_genes, 2)
    v_emb = np.stack(
        [adata.layers[velocity_key], adata.layers[velocity_key + "_unspliced"]],
        axis=2,
    )  # (n_cells, n_genes, 2)

    # remove nan genes
    nan_genes = np.isnan(v_emb).any(axis=(0, 2)) | np.isnan(x_emb).any(axis=(0, 2))
    x_emb = x_emb[:, ~nan_genes, :]
    v_emb = v_emb[:, ~nan_genes, :]

    if np.max(adata.layers[velocity_key + "_unspliced"]) <= 1e-5:
        # for the case there is no actual unspliced estimates, set unspliced emb to 0
        x_emb[:, :, 1] = 0

    for u, v in cluster_edges:
        assert u in adata.obs[cluster_key].cat.categories, f"cluster {u} not found"
        assert v in adata.obs[cluster_key].cat.categories, f"cluster {v} not found"
        sel = adata.obs[cluster_key] == u
        nbs = adata.uns["neighbors"]["indices"][sel]  # [n * 30]

        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, cluster_key), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue

            x_diff = x_emb[nodes] - x_pos

            x_diff = normalize(x_diff.reshape(-1, x_diff.shape[-1]), axis=1).reshape(
                x_diff.shape
            )  # (n_nodes, n_genes, 2)
            x_vel = normalize(x_vel, axis=1)  # (n_genes, 2)
            dir_scores = np.sum(x_diff * x_vel, axis=-1)  # (n_nodes, n_genes)

            type_score.append(np.mean(dir_scores, axis=0))  # (n_genes,)

        type_score = np.stack(type_score, axis=0)  # (n_cells, n_genes)
        gene_wise_scores_on_boundaries[(u, v)] = np.mean(type_score, axis=0)

    gene_wise_overall_scores = np.mean(
        np.stack(gene_wise_scores_on_boundaries.values(), axis=0), axis=0
    )  # (n_genes,)

    if inplace:
        adata.uns["genewise_raw_direction_scores"] = gene_wise_scores_on_boundaries
        adata.var["gene_direction_scores"] = np.full(adata.shape[1], np.nan)
        adata.var["gene_direction_scores"][~nan_genes] = gene_wise_overall_scores
        logg.info(f"added 'genewise_raw_direction_scores' (adata.uns)")
        logg.info(f"added 'gene_direction_scores' (adata.var)")

    return gene_wise_scores_on_boundaries, gene_wise_overall_scores


def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type

    Args:
        adata (Anndata): Anndata object.
        nodes (list): Indexes for cells
        target (str): Cluster name.
        k_cluster (str): Cluster key in adata.obs dataframe
    Returns:
        list: Selected cells.
    """
    return nodes[adata.obs[k_cluster][nodes].values == target]


def continuity_confidence(
    adata: AnnData,
    trainer: Optional["Trainer"] = None,
    copy: bool = False,
    threshold_quantile: float = 0.05,
) -> Optional[AnnData]:
    """
    Compute confidence on a whole anndata object.

    Args:
        adata (`anndata.AnnData`): Annotated data matrix.
        trainer (`Trainer`): The trained deepvelo trainer that contains
            computed raw continuity values.
        copy (bool, optional): Return a copy instead of writing to adata.

    Returns:
        `anndata.AnnData` or `None`

    Updates in `adata`:
        continuity_mse: `.layers`, mean squared error from the continuity assumption.
        continuity_relative_error: `.layers`, relative error from the continuity assumption.
        cell_continuity: `.obs`, cell-wise continuity score.
        gene_continuity: `.var`, gene-wise continuity score.
        genn_corr: `.var`, gene-wise correlation from the pearson correlation objective.
    """

    if copy:
        adata = adata.copy()

    if trainer is None:
        raise NotImplementedError("trainer is not provided.")

    if not hasattr(trainer, "confidence"):
        raise ValueError("trainer provided, but trainer.confidence is not computed.")

    adata.layers["continuity_mse"] = trainer.confidence["mse"]
    logg.info("added 'continuity_mse' (adata.layers)")

    # # raw
    # cell_continuity_mse = trainer.confidence["mse"].mean(axis=1)
    # gene_continuity_mse = trainer.confidence["mse"].mean(axis=0)

    # relative over average
    adata.layers["continuity_relative_error"] = np.sqrt(trainer.confidence["mse"])
    logg.info("added 'continuity_relative_error' (adata.layers)")
    cell_continuity_mse = adata.layers["continuity_relative_error"].mean(axis=1) / (
        adata.layers["Ms"].mean(axis=1) + 1e-6
    )
    gene_continuity_mse = adata.layers["continuity_relative_error"].mean(axis=0) / (
        adata.layers["Ms"].mean(axis=0) + 1e-6
    )

    # # average over relative
    # adata.layers["continuity_relative_error"] = (
    #     np.sqrt(trainer.confidence["mse"]) + np.sqrt(trainer.confidence["mse"]).mean()
    # ) / (adata.layers["Ms"] + adata.layers["Ms"].mean())  # to fix when random guessing using the expression prior, how well it can get
    # # adata.layers["continuity_relative_error"] = np.exp(np.log1p(np.sqrt(trainer.confidence["mse"])) - np.log1p(adata.layers["Ms"]))
    # cell_continuity_mse = adata.layers["continuity_relative_error"].mean(axis=1)
    # gene_continuity_mse = adata.layers["continuity_relative_error"].mean(axis=0)

    gene_corr = trainer.confidence["corr"]
    adata.obs["cell_continuity"] = 1 - np.tanh(cell_continuity_mse)
    adata.var["gene_continuity"] = 1 - np.tanh(gene_continuity_mse)
    adata.var["gene_corr"] = gene_corr
    logg.info("added 'cell_continuity' (adata.obs)")
    logg.info("added 'gene_continuity' (adata.var)")
    logg.info("added 'gene_corr' (adata.var)")

    if isinstance(threshold_quantile, float) and threshold_quantile > 0:
        continuity_threshold = adata.var["gene_continuity"].quantile(threshold_quantile)
        logg.info(f"continuity threshold: {continuity_threshold}")
        adata.var["continuous_gene"] = (
            adata.var["gene_continuity"] > continuity_threshold
        )
        logg.info("added 'continuous_gene' (adata.var)")

        # corr_threshold = np.quantile(gene_corr, threshold_quantile)
        # corr_threshold = max(corr_threshold, 0.0)
        corr_threshold = 0.0
        logg.info(f"correlation threshold: {corr_threshold}")
        adata.var["correlated_gene"] = gene_corr > corr_threshold
        logg.info("added 'correlated_gene' (adata.var)")

    return adata if copy else None
