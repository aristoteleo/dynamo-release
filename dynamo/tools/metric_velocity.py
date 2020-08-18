import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import issparse, csr_matrix
from sklearn.neighbors import NearestNeighbors
from .connectivity import umap_conn_indices_dist_embedding, mnn_from_list
from .utils import (
    get_finite_inds,
    inverse_norm,
    einsum_correlation,
    fetch_X_data,
)


def cell_wise_confidence(adata,
                         X_data=None,
                         V_data=None,
                         ekey="M_s",
                         vkey="velocity_S",
                         neighbors_from_basis=False,
                         method="jaccard"):
    """ Calculate the cell-wise velocity confidence metric.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        X_data: 'np.ndarray' or `sp.csr_matrix` or None (optional, default `None`)
            The expression states of single cells (or expression states in reduced dimension, like pca, of single cells)
        V_data: 'np.ndarray' or `sp.csr_matrix` or None (optional, default `None`)
            The RNA velocity of single cells (or velocity estimates projected to reduced dimension, like pca, of single
            cells). Note that X, V_mat need to have the exact dimensionalities.
        ekey: `str` (optional, default `M_s`)
            The dictionary key that corresponds to the gene expression in the layer attribute. By default, it is the
            smoothed expression `M_s`.
        vkey: 'str' (optional, default `velocity_S`)
            The dictionary key that corresponds to the estimated velocity values in layers attribute.
        neighbors_from_basis: `bool` (optional, default `False`)
            Whether to construct nearest neighbors from low dimensional space as defined by the `basis`, instead of using
            that calculated during UMAP process.
        method: `str` (optional, default `jaccard`)
            Which method will be used for calculating the cell wise velocity confidence metric.
            By default it uses
            `jaccard` index, which measures how well each velocity vector meets the geometric constraints defined by the
            local neighborhood structure. Jaccard index is calculated as the fraction of the number of the intersected
            set of nearest neighbors from each cell at current expression state (X) and that from the future expression
            state (X + V) over the number of the union of these two sets. The `cosine` or `correlation` method is similar
            to that used by scVelo (https://github.com/theislab/scvelo).

    Returns
    -------
        Adata: :class:`~anndata.AnnData`
            Returns an updated `~anndata.AnnData` with `.obs.confidence` as the cell-wise velocity confidence.
    """

    if ekey == "X":
        X, V = (adata.X if X_data is None else X_data, adata.layers[vkey] if V_data is None else V_data)
        norm_method = adata.uns["pp_norm_method"].copy()
        adata.uns["pp_norm_method"] = 'log1p'
        X = inverse_norm(adata, X) if X_data is None else X_data
        adata.uns["pp_norm_method"] = norm_method
    else:
        X, V = (adata.layers[ekey] if X_data is None else X_data, adata.layers[vkey] if V_data is None else V_data)
        X = inverse_norm(adata, X) if X_data is None else X_data

    if not neighbors_from_basis:
        n_neigh, X_neighbors = (
            adata.uns["neighbors"]["params"]["n_neighbors"],
            adata.obsp["connectivities"],
        )
    else:
        n_neigh = 30

        if X.shape[0] > 200000 and X.shape[1] > 2: 
            from pynndescent import NNDescent

            nbrs = NNDescent(X, metric='euclidean', n_neighbors=n_neigh + 1, n_jobs=-1, random_state=19491001)
            nbrs_idx, dist = nbrs.query(X, k=n_neigh + 1)
        else:
            alg = 'ball_tree' if X.shape[1] > 10 else 'kd_tree'
            nbrs = NearestNeighbors(n_neighbors=n_neigh + 1, algorithm=alg, n_jobs=-1).fit(X)
            dist, nbrs_idx = nbrs.kneighbors(X)

        row = np.repeat(nbrs_idx[:, 0], n_neigh)
        col = nbrs_idx[:, 1:].flatten()
        X_neighbors = csr_matrix((np.repeat(1, len(col)), (row, col)), shape=(adata.n_obs, adata.n_obs))

    n_neigh = n_neigh[0] if type(n_neigh) == np.ndarray else n_neigh
    n_pca_components = adata.obsm["X"].shape[1]

    finite_inds = get_finite_inds(V, 0)
    X, V = X[:, finite_inds], V[:, finite_inds]
    if method == "jaccard":
        jac, _, _ = jaccard(X, V, n_pca_components, n_neigh, X_neighbors)
        confidence = jac

    elif method == "hybrid":
        # this is inspired from the locality preservation paper
        jac, intersect_, _ = jaccard(X, V, n_pca_components, n_neigh, X_neighbors)

        confidence = np.zeros(adata.n_obs)
        for i in tqdm(range(adata.n_obs), desc='calculating hybrid method (jaccard + consensus) based cell wise confidence'):
            neigh_ids = np.where(intersect_[i].A)[0] if issparse(intersect_) else np.where(intersect_[i])[0]
            confidence[i] = jac[i] * np.mean(
                [consensus(V[i].A.flatten(), V[j].A.flatten()) for j in neigh_ids]
            )  if issparse(V) else jac[i] * np.mean(
                [consensus(V[i].flatten(), V[j].flatten()) for j in neigh_ids]
            )

    elif method == "cosine":
        indices = adata.uns["neighbors"]["indices"]
        confidence = np.zeros(adata.n_obs)
        for i in tqdm(range(adata.n_obs), desc='calculating cosine based cell wise confidence'):
            neigh_ids = indices[i]
            confidence[i] = np.mean(
                [einsum_correlation(V[i].A, V[j].A.flatten(), type="cosine")[0, 0] for j in neigh_ids]
            ) if issparse(V) else np.mean(
                [einsum_correlation(V[i][None, :], V[j].flatten(), type="cosine")[0, 0] for j in neigh_ids]
            )

    elif method == "consensus":
        indices = adata.uns["neighbors"]["indices"]
        confidence = np.zeros(adata.n_obs)
        for i in tqdm(range(adata.n_obs), desc='calculating consensus based cell wise confidence'):
            neigh_ids = indices[i]
            confidence[i] = np.mean(
                [consensus(V[i].A.flatten(), V[j].A.flatten()) for j in neigh_ids]
            ) if issparse(V) else np.mean(
                [consensus(V[i], V[j].flatten()) for j in neigh_ids]
            )

    elif method == "correlation":
        # this is equivalent to scVelo
        indices = adata.uns["neighbors"]["indices"]
        confidence = np.zeros(adata.n_obs)
        for i in tqdm(range(adata.n_obs), desc='calculating correlation based cell wise confidence'):
            neigh_ids = indices[i]
            confidence[i] = np.mean(
                [einsum_correlation(V[i].A, V[j].A.flatten(), type="pearson")[0, 0] for j in neigh_ids]
            ) if issparse(V) else np.mean(
                [einsum_correlation(V[i][None, :], V[j].flatten(), type="pearson")[0, 0] for j in neigh_ids]
            )

    elif method == "divergence":
        pass

    else:
        raise Exception(
            "The input {} method for cell-wise velocity confidence calculation is not implemented"
            " yet".format(method)
        )

    adata.obs[method + "_velocity_confidence"] = confidence

    return adata


def jaccard(X, V, n_pca_components, n_neigh, X_neighbors):
    from sklearn.decomposition import TruncatedSVD

    transformer = TruncatedSVD(n_components=n_pca_components + 1, random_state=0)
    Xt = X + V
    if issparse(Xt):
        Xt.data[Xt.data < 0] = 0
        Xt.data = np.log2(Xt.data + 1)
    else:
        Xt = np.log2(Xt + 1)
    X_fit = transformer.fit(Xt)
    Xt_pca = X_fit.transform(Xt)[:, 1:]

    V_neighbors, _, _, _ = umap_conn_indices_dist_embedding(
        Xt_pca, n_neighbors=n_neigh, return_mapper=False
    )
    X_neighbors_, V_neighbors_ = (
        X_neighbors.dot(X_neighbors),
        V_neighbors.dot(V_neighbors),
    )
    union_ = X_neighbors_ + V_neighbors_ > 0
    intersect_ = mnn_from_list([X_neighbors_, V_neighbors_]) > 0

    jaccard = (
        (intersect_.sum(1) / union_.sum(1)).A1
        if issparse(X)
        else intersect_.sum(1) / union_.sum(1)
    )

    return jaccard, intersect_, union_


def consensus(x, y):
    x_norm, y_norm = np.linalg.norm(x), np.linalg.norm(y)
    consensus = einsum_correlation(x[None, :], y, type="cosine")[0, 0] * \
                np.min([x_norm, y_norm]) / np.max([x_norm, y_norm])

    return consensus


def gene_wise_confidence(adata,
                         group,
                         lineage_dict=None,
                         genes=None,
                         ekey='M_s',
                         vkey='velocity_S',
                         X_data=None,
                         V_data=None,
                         V_threshold=1,
                         ):
    """Diagnostic measure to identify genes contributed to "wrong" directionality of the vector flow.

    In some scenarios, you may find unexpected "wrong vector backflow" from your dynamo analysis, in order to diagnose
    those cases, we can identify those genes showing up in the wrong phase portrait position. Then we nay remove those
    identified genes to "correct" velocity vectors. This requires us to give some priors about what progenitor and
    terminal cell types are. The rationale behind this basically boils down to understanding the following two
    scenarios:

    1). if the progenitor’s expression is low, starting from time point 0, cells should start to increase expression.
    There must be progenitors that are above the steady-state line. However, if most of the progenitors are laying below
    the line (indicated by the red cells), we will have negative velocity and this will lead to reversed vector flow.

    2). if progenitors start from high expression, starting from time point 0, cells should start to decrease expression.
    There must be progenitors that are below the steady-state line. However, if most of the progenitors are laying above
    the steady state line, we will have positive velocity and this will lead to reversed vector flow.

    The same rationale can be applied to the mature cell states.

    Thus, we design an algorithm to access the confidence of each gene obeying the above two constraints:
    We first check for whether a gene should be in the induction or repression phase from each progenitor to each
    terminal cell states (based on the shift of the median gene expression between these two states). If it is in
    induction phase, cells should show mostly at >= small negative velocity; otherwise <= small negative velocity.
    1 - ratio of cells with velocity pass those threshold (defined by `V_threshold`) in each state is then defined as a
    velocity confidence measure.

    Note that, this heuristic method requires you provide meaningful `progenitors_groups` and `mature_cells_groups`. In
    particular, the progentitor groups should in principle have cell going out (transcriptomically) while mature groups
    should end up in a different expression state and there are intermediate cells going to the dead end cells in the
    each terminal group (or most terminal groups).

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        group: `str`
            The column key/name that identifies the cell state grouping information of cells. This will be used for
            calculating gene-wise confidence score in each cell state.
        lineage_dict: `dict`
            A dictionary describes lineage priors. Keys corresponds to the group name from `group` that corresponding
            to the state of one progenitor type while values correspond to the group names from `group` that
            corresponding to the states of one or multiple terminal cell states. The best practice for determining
            terminal cell states are those fully functional cells instead of intermediate cell states. Note that in
            python a dictionary key cannot be a list, so if you have two progenitor types converge into one terminal
            cell state, you need to create two records each with the same terminal cell as value but different progenitor
            as the key. Value can be either a string for one cell group or a list of string for multiple cell groups.
        genes: `list` or None (default: `None`)
            The list of genes that will be used to gene-wise confidence score calculation. If `None`, all genes that go
            through velocity estimation will be used.
        ekey: `str` or None (default: `M_s`)
            The layer that will be used to retrieve data for identifying the gene is in induction or repression phase at
            each cell state. If `None`, .X is used.
        vkey: `str` or None (default: `velocity_S`)
            The layer that will be used to retrieve velocity data for calculating gene-wise confidence. If `None`,
            `velocity_S` is used.
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for identifying the gene is in induction or repression phase at
            each cell state directly
        V_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for calculating gene-wise confidence directly.
        V_threshold: `float` (default: `1`)
            The threshold of velocity to calculate the gene wise confidence.

    Returns
    -------
        A updated adata object with a new `gene_wise_confidence` key in .uns, which contains gene-wise confidence score
        in each cell state. .var will also be updated with `avg_prog_confidence` and `avg_mature_confidence` key which
        correspond to the average gene wise confidence in the progenitor state or the mature cell state.
    """

    if X_data is None:
        genes, X_data = fetch_X_data(adata, genes, ekey)
    else:
        if genes is None or len(genes) != X_data.shape[1]:
            raise ValueError(f"When providing X_data, a list of genes name that corresponds to the columns of X_data "
                             f"must be provided")
    if V_data is None:
        genes, V_data = fetch_X_data(adata, genes, vkey)
    else:
        if genes is None or len(genes) != X_data.shape[1]:
            raise ValueError(f"When providing V_data, a list of genes name that corresponds to the columns of X_data "
                             f"must be provided")

    sparse, sparse_v = issparse(X_data), issparse(V_data)

    confidence = []
    for i_gene, gene in tqdm(enumerate(genes), desc="calculating gene velocity vectors confidence based on phase "
                                                    "portrait location with priors of progenitor/mature cell types"):
        all_vals = X_data[:, i_gene].A if sparse else X_data[:, i_gene]
        all_vals_v = V_data[:, i_gene].A if sparse_v else V_data[:, i_gene]

        for progenitors_groups, mature_cells_groups in lineage_dict.items():
            progenitors_groups = [progenitors_groups]
            if type(mature_cells_groups) is str:
                mature_cells_groups = [mature_cells_groups]

            for i, progenitor in enumerate(progenitors_groups):
                prog_vals = all_vals[adata.obs[group] == progenitor]
                prog_vals_v = all_vals_v[adata.obs[group] == progenitor]
                threshold_val = np.percentile(abs(all_vals_v), V_threshold)

                for j, mature in enumerate(mature_cells_groups):
                    mature_vals = all_vals[adata.obs[group] == mature]
                    mature_vals_v = all_vals_v[adata.obs[group] == mature]

                    if np.nanmedian(prog_vals) - np.nanmedian(mature_vals) > 0:
                        # repression phase (bottom curve -- phase curve below the linear line indicates steady states)
                        prog_confidence = 1 - sum(prog_vals_v > - threshold_val)[0] / len(
                            prog_vals_v)  # most cells should downregulate / ss
                        mature_confidence = 1 - sum(mature_vals_v > - threshold_val)[0] / len(
                            mature_vals_v)  # most cell should downregulate / ss
                    else:
                        # induction phase (upper curve -- phase curve above the linear line indicates steady states)
                        prog_confidence = 1 - sum(prog_vals_v < threshold_val)[0] / len(
                            prog_vals_v)  # most cells should upregulate / ss
                        mature_confidence = 1 - sum(mature_vals_v < threshold_val)[0] / len(
                            mature_vals_v)  # most cell should upregulate / ss

                    confidence.append((gene, progenitor, mature, prog_confidence, mature_confidence))

    confidence = pd.DataFrame(confidence,
                      columns=['gene', 'progenitor', 'mature', 'prog_confidence', 'mature_confidence'])
    confidence.astype(dtype={"prog_confidence": "float64",
                             "prog_confidence": "float64"})
    adata.var['avg_prog_confidence'], adata.var['avg_mature_confidence'] = np.nan, np.nan
    avg = confidence.groupby('gene')['prog_confidence', 'mature_confidence'].mean()
    avg = avg.reset_index().set_index('gene')
    adata.var.loc[genes, 'avg_prog_confidence'] = avg.loc[genes, 'prog_confidence']
    adata.var.loc[genes, 'avg_mature_confidence'] = avg.loc[genes, 'mature_confidence']

    adata.uns['gene_wise_confidence'] = confidence

