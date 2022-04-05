import warnings
from typing import Union

import anndata
import numpy as np
import scipy
import scipy as scp
import scipy.sparse as sp
from numba import jit
from sklearn.decomposition import PCA
from sklearn.utils import sparsefuncs

# dynamo logger related
from ..dynamo_logger import (
    LoggerManager,
    main_critical,
    main_exception,
    main_info,
    main_tqdm,
    main_warning,
)
from ..utils import areinstance
from .connectivity import _gen_neighbor_keys, adj_to_knn, check_and_recompute_neighbors
from .dimension_reduction import reduceDimension
from .graph_calculus import calc_gaussian_weight, fp_operator, graphize_velocity
from .Markov import ContinuousTimeMarkovChain, KernelMarkovChain, velocity_on_grid
from .metric_velocity import gene_wise_confidence
from .utils import (
    einsum_correlation,
    get_ekey_vkey_from_adata,
    get_finite_inds,
    get_mapper_inverse,
    get_neighbor_indices,
    index_gene,
    log1p_,
    norm,
    set_transition_genes,
    split_velocity_graph,
    update_dict,
)


def cell_velocities(
    adata: anndata.AnnData,
    ekey: Union[str, None] = None,
    vkey: Union[str, None] = None,
    X: Union[np.array, scipy.sparse.csr_matrix, None] = None,
    V: Union[np.array, scipy.sparse.csr_matrix, None] = None,
    X_embedding: Union[str, None] = None,
    transition_matrix: Union[np.array, scipy.sparse.csr_matrix, None] = None,
    use_mnn: bool = False,
    n_pca_components: Union[int, None] = None,
    transition_genes: Union[str, list, None] = None,
    min_r2: Union[float, None] = None,
    min_alpha: Union[float, None] = None,
    min_gamma: Union[float, None] = None,
    min_delta: Union[float, None] = None,
    basis: str = "umap",
    neighbor_key_prefix: str = "",
    adj_key: str = "distances",
    add_transition_key: str = None,
    add_velocity_key: str = None,
    n_neighbors: int = 30,
    method: str = "pearson",
    neg_cells_trick: bool = True,
    calc_rnd_vel: bool = False,
    xy_grid_nums: tuple = (50, 50),
    correct_density: bool = True,
    scale: bool = True,
    sample_fraction: Union[float, None] = None,
    random_seed: int = 19491001,
    enforce: bool = False,
    preserve_len: bool = False,
    **kernel_kwargs,
) -> anndata.AnnData:
    """Project high dimensional velocity vectors onto given low dimensional embeddings,
    and/or compute cell transition probabilities.

    When method='kmc', the Itô kernel is used which not only considers the correlation between the vector from any cell
    to its nearest neighbors and its velocity vector but also the corresponding distances. We expect this new kernel
    will enable us to visualize more intricate vector flow or steady states in low dimension. We also expect it will
    improve the calculation of the stationary distribution or source states of sampled cells. The original
    "correlation/cosine" velocity projection method is also supported. Kernels based on the reconstructed velocity field
    is also possible.

    With the `key` argument, `cell_velocities` can be called by `cell_accelerations` or `cell_curvature` to calculate
    RNA acceleration/curvature vector for each cell.

    Arguments
    ---------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        ekey: str or None (optional, default None)
            The dictionary key that corresponds to the gene expression in the layer attribute. By default, ekey and vkey
            will be automatically detected from the adata object.
        vkey: str or None (optional, default None)
            The dictionary key that corresponds to the estimated velocity values in the layers attribute.
        X: :class:`~numpy.ndarray` or :class:`~scipy.sparse.csr_matrix` or None (optional, default `None`)
            The expression states of single cells (or expression states in reduced dimension, like pca, of single cells)
        V: :class:`~numpy.ndarray` or :class:`~scipy.sparse.csr_matrix` or None (optional, default `None`)
            The RNA velocity of single cells (or velocity estimates projected to reduced dimension, like pca, of single
            cells). Note that X, V need to have the exact dimensionalities.
        X_embedding: str or None (optional, default None)
            The low expression reduced space (pca, umap, tsne, etc.) of single cells that RNA velocity will be projected
            onto. Note X_embedding, X and V has to have the same cell/sample dimension and X_embedding should have
            less feature dimension comparing that of X or V.
        use_mnn: bool (optional, default False)
            Whether to use mutual nearest neighbors for projecting the high dimensional velocity vectors. By default, we
            don't use the mutual nearest neighbors. Mutual nearest neighbors are calculated from nearest neighbors
            across different layers, which which accounts for cases where, for example, the cells from spliced
            expression may be nearest neighbors but far from nearest neighbors on unspliced data. Using mnn assumes your
            data from different layers are reliable (otherwise it will destroy real signals).
        n_pca_components: int (optional, default None)
            The number of pca components to project the high dimensional X, V before calculating transition matrix for
            velocity visualization. By default it is None and if method is `kmc`, n_pca_components will be reset to 30;
            otherwise use all high dimensional data for velocity projection.
        transition_genes: str, list, or None (optional, default None)
            The set of genes used for projection of hign dimensional velocity vectors.
            If None, transition genes are determined based on the R2 of linear regression on phase planes.
            The argument can be either a dictionary key of .var, a list of gene names, or a list of booleans
            of length .n_vars.
        min_r2: float or None (optional, default None)
            The minimal value of r-squared of the parameter fits for selecting transition genes.
        min_alpha: float or None (optional, default None)
            The minimal value of alpha kinetic parameter for selecting transition genes.
        min_gamma: float or None (optional, default None)
            The minimal value of gamma kinetic parameter for selecting transition genes.
        min_delta: float or None (optional, default None)
            The minimal value of delta kinetic parameter for selecting transition genes.
        basis: str (optional, default `umap`)
            The dictionary key that corresponds to the reduced dimension in `.obsm` attribute. Can be `X_spliced_umap`
            or `X_total_umap`, etc.
        neighbor_key_prefix: str (optional, default `neighbors`)
            The dictionary key prefix in .uns. Connectivity and distance matrix keys are also generate with this prefix in adata.obsp.
        adj_key: str (optional, default `distances`)
            The dictionary key for the adjacency matrix of the nearest neighbor graph in .obsp.
        add_transition_key: str or None (default: None)
            The dictionary key that will be used for storing the transition matrix in .obsp.
        add_velocity_key: str or None (default: None)
            The dictionary key that will be used for storing the low dimensional velocity projection matrix in .obsm.
        method: str (optional, default `pearson`)
            The method to calculate the transition matrix and project high dimensional vector to low dimension, either
            `kmc`, `fp`, `cosine`, `pearson`, or `transform`. "kmc" is our new approach to learn the transition matrix
            via diffusion approximation or an Itô kernel. "cosine" or "pearson" are the methods used in the original RNA
            velocity paper or the scvelo paper (Note that scVelo implementation actually centers both dX and V, so its
            cosine kernel is equivalent to pearson correlation kernel but we also provide the raw cosine kernel). "kmc"
            option is arguable better than "correlation" or "cosine" as it not only considers the correlation but also
            the distance of the nearest neighbors to the high dimensional velocity vector. Finally, the "transform"
            method uses umap's transform method to transform new data points to the UMAP space. "transform" method is
            NOT recommended. Kernels that are based on the reconstructed vector field in high dimension is also
            possible.
        neg_cells_trick: bool (optional, default True)
            Whether we should handle cells having negative correlations in gene expression difference with high
            dimensional velocity vector separately. This option was borrowed from scVelo package
            (https://github.com/theislab/scvelo) and use in conjunction with "pearson" and "cosine" kernel. Not required
            if method is set to be "kmc".
        calc_rnd_vel: bool (default: False)
            A logic flag to determine whether we will calculate the random velocity vectors which can be plotted
            downstream as a negative control and used to adjust the quiver scale of the velocity field.
        xy_grid_nums: tuple (default: (50, 50)).
            A tuple of number of grids on each dimension.
        correct_density: bool (default: True)
            Whether to correct density when calculating the markov transition matrix.
        scale: bool (default: False)
            Whether to scale velocity when calculating the markov transition matrix, applicable to the `kmc` kernel.
        sample_fraction: None or float (default: None)
            The downsampled fraction of kNN for the purpose of acceleration, applicable to the `kmc` kernel.
        random_seed: int (default: 19491001)
            The random seed for numba to ensure consistency of the random velocity vectors. Default value 19491001 is a
            special day for those who care.
        key: str or None (default: None)
            The prefix key that will be prefixed to the keys for storing calculated transition matrix, projection
            vectors, etc.
        preserve_len: bool (default: False)
            Whether to preserve the length of high dimension vector length. When set to be True, the length  of low
            dimension projected vector will be proportionally scaled to that of the high dimensional vector.
        enforce: bool (default: False)
            Whether to enforce 1) redefining use_for_transition column in obs attribute; However this is NOT executed if
                                    the argument 'transition_genes' is not None.
                               2) recalculation of the transition matrix.
        kernel_kwargs: dict
            A dictionary of paramters that will be passed to the kernel for constructing the transition matrix.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            Returns an updated :class:`~anndata.AnnData` with projected velocity vectors, and a cell transition matrix
            calculated using either the Itô kernel method or similar methods from (La Manno et al. 2018).
    """
    conn_key, dist_key, neighbor_key = _gen_neighbor_keys(neighbor_key_prefix)
    mapper_r = get_mapper_inverse()
    layer = mapper_r[ekey] if (ekey is not None and ekey in mapper_r.keys()) else ekey
    ekey, vkey, layer = get_ekey_vkey_from_adata(adata) if (ekey is None or vkey is None) else (ekey, vkey, layer)

    if calc_rnd_vel:
        numba_random_seed(random_seed)

    if neighbor_key is not None and neighbor_key in adata.uns.keys() and "indices" in adata.uns[neighbor_key]:
        check_and_recompute_neighbors(adata, result_prefix=neighbor_key_prefix)
        # use neighbor indices in neighbor_key (if available) first for the sake of performance.
        indices = adata.uns[neighbor_key]["indices"]
        if type(indices) is not np.ndarray:
            indices = np.array(indices)

        # simple case: the input is a knn graph
        if len(indices.shape) > 1:
            indices = indices[:, :n_neighbors]
        # general case
        else:
            idx = np.ones((len(indices), n_neighbors)) * np.nan
            for i, nbr in enumerate(indices):
                idx[i, : len(nbr)] = nbr
            indices = idx
            if np.any(np.isnan(indices)):
                main_warning("Resulting knn index matrix contains NaN. Check if n_neighbors is too large.")

    elif adj_key is not None and adj_key in adata.obsp.keys():
        if use_mnn:
            neighbors = adata.uns["mnn"]
            indices, _ = adj_to_knn(neighbors, adata.uns["neighbors"]["indices"].shape[1])
            indices = indices[:, 1:]
        else:
            knn_indices, _ = adj_to_knn(adata.obsp[adj_key], n_neighbors)
            # knn_adj = knn_to_adj(knn_indices, knn_dists)

            # user wouldn't expect such a function to change the neighborhood info...
            # consider writing them into a new item, or do this in connectivity.neighbors.

            #    adata.uns["neighbors"]["indices"], adata.obsp["distances"] = knn_indices, knn_adj
            # dist, indices = (
            #    adata.obsp["distances"],
            #    adata.uns["neighbors"]["indices"],
            # )
            # indices, dist = indices[:, 1:], dist[:, 1:]
            indices = knn_indices[:, 1:]
    else:
        raise Exception(
            f"Neighborhood info '{adj_key}' is missing in the provided anndata object."
            "Run `dyn.tl.reduceDimension` or `dyn.tl.neighbors` first."
        )

    if X is None and V is None:
        if transition_genes is None:
            if "use_for_transition" not in adata.var.keys() or enforce:
                use_for_dynamics = True if "use_for_dynamics" in adata.var.keys() else False
                adata = set_transition_genes(
                    adata,
                    vkey=vkey,
                    min_r2=min_r2,
                    use_for_dynamics=use_for_dynamics,
                    min_alpha=min_alpha,
                    min_gamma=min_gamma,
                    min_delta=min_delta,
                )
            transition_genes = adata.var_names[adata.var.use_for_transition.values]
        else:
            if not enforce:
                main_warning(
                    "A new set of transition genes is used, but because enforce=False, "
                    "the transition matrix might not be recalculated if it is found in .obsp."
                )
            dynamics_genes = (
                adata.var.use_for_dynamics
                if "use_for_dynamics" in adata.var.keys()
                else np.ones(adata.n_vars, dtype=bool)
            )
            if type(transition_genes) is str:
                transition_genes = adata.var[transition_genes].to_list()
                transition_genes = np.logical_and(transition_genes, dynamics_genes.to_list())
            elif areinstance(transition_genes, str):
                transition_genes = adata.var_names[dynamics_genes].intersection(transition_genes).to_list()
            elif areinstance(transition_genes, bool) or areinstance(transition_genes, np.bool_):
                transition_genes = np.array(transition_genes)
                transition_genes = np.logical_and(transition_genes, dynamics_genes.to_list())
            else:
                raise TypeError(
                    "transition genes should either be a key of adata.var, an array of gene names, or of booleans."
                )
            if len(transition_genes) < 1:
                raise ValueError(
                    "None of the transition genes provided has velocity values. (or `var.use_for_dynamics` is `False`)."
                )

            adata.var["use_for_transition"] = False
            if type(transition_genes[0]) == bool:
                adata.var.use_for_transition = transition_genes
            else:
                adata.var.loc[transition_genes, "use_for_transition"] = True

    # X = adata[:, transition_genes].layers[ekey] if X is None else X
    X = index_gene(adata, adata.layers[ekey], transition_genes) if X is None else X

    V = (
        (
            # adata[:, transition_genes].layers[vkey]
            index_gene(adata, adata.layers[vkey], transition_genes)
            if vkey in adata.layers.keys()
            else None
        )
        if V is None
        else V
    )

    if X.shape != V.shape:
        raise Exception("X and V do not have the same number of dimensions.")

    if X_embedding is None:
        has_splicing, has_labeling = (
            adata.uns["dynamics"]["has_splicing"],
            adata.uns["dynamics"]["has_labeling"],
        )
        if has_splicing and has_labeling:
            main_warning(
                "\nYour data has both labeling / splicing data, please ensuring using the right `basis` "
                "({basis}):"
                "\n   when using `velocity_S`, please use basis based on X_spliced data;"
                "\n   when using `velocity_T, please use basis based X_total. "
                "\nIf not sure the data in adata.X, you may need to set `basis='X_spliced_umap'`"
                "(`basis='X_total_umap'`) when using `velocity_S` (`velocity_T`). "
                ""
            )

        if "_" in basis and any([i in basis for i in ["X_", "spliced_", "unspliced_", "new_", "total"]]):
            basis_layer, basis = basis.rsplit("_", 1)
            reduceDimension(adata, layer=basis_layer, reduction_method=basis)
            X_embedding = adata.obsm[basis]
        else:
            if vkey in ["velocity_S", "velocity_T"]:
                X_embedding = adata.obsm["X_" + basis]
            else:
                reduceDimension(adata, layer=layer, reduction_method=basis)
                X_embedding = adata.obsm[layer + "_" + basis]

    if X.shape[0] != X_embedding.shape[0]:
        raise Exception("X and X_embedding do not have the same number of samples.")
    if X.shape[1] < X_embedding.shape[1]:
        raise Exception(
            "The number of dimensions of X is smaller than that of the embedding. Try lower the min_r2, "
            "min_gamma thresholds."
        )

    V = V.A if sp.issparse(V) else V
    X = X.A if sp.issparse(X) else X
    finite_inds = get_finite_inds(V)
    X, V = X[:, finite_inds], V[:, finite_inds]

    if sum(finite_inds) != X.shape[0]:
        main_info(f"{X.shape[1] - sum(finite_inds)} genes are removed because of nan velocity values.")
        if transition_genes is not None:  # if X, V is provided by the user, transition_genes will be None
            adata.var.loc[np.array(transition_genes)[~finite_inds], "use_for_transition"] = False

    if finite_inds.sum() < 5 and len(finite_inds) > 100:
        raise Exception(
            f"there are only {finite_inds.sum()} genes have finite velocity values. "
            f"Please make sure the {vkey} is correctly calculated! And if you run kinetic parameters "
            "estimation for each cell-group via `group` argument, make sure all groups have sufficient "
            "number of cells, e.g. 50 cells at least. Otherwise some cells may have NaN values for all "
            "genes."
        )

    if method == "kmc" and n_pca_components is None:
        n_pca_components = 30
    if n_pca_components is not None:
        X = log1p_(adata, X)
        X_plus_V = log1p_(adata, X + V)
        if "velocity_pca_fit" not in adata.uns_keys() or type(adata.uns["velocity_pca_fit"]) == str:
            pca_monocle = PCA(
                n_components=min(n_pca_components, X.shape[1] - 1),
                svd_solver="arpack",
                random_state=0,
            )
            pca_fit = pca_monocle.fit(X)
            X_pca = pca_fit.transform(X)

            adata.uns["velocity_pca_fit"] = pca_fit
            adata.uns["velocity_PCs"] = pca_fit.components_.T
            adata.obsm["X_velocity_pca"] = X_pca

        X_pca, _, pca_fit = (
            adata.obsm["X_velocity_pca"],
            adata.uns["velocity_PCs"],
            adata.uns["velocity_pca_fit"],
        )

        Y_pca = pca_fit.transform(X_plus_V)
        V_pca = Y_pca - X_pca

        adata.obsm["velocity_pca_raw"] = V_pca
        X, V = X_pca[:, :n_pca_components], V_pca[:, :n_pca_components]

    # add both source and sink distribution
    if method == "kmc":
        if method + "_transition_matrix" in adata.obsp.keys() and not enforce:
            T = adata.obsp[method + "_transition_matrix"] if transition_matrix is None else transition_matrix
            kmc = KernelMarkovChain(P=T)
        else:
            kmc = KernelMarkovChain()
        kmc_args = {
            "n_recurse_neighbors": 2,
            "M_diff": 2,
            "epsilon": None,
            "adaptive_local_kernel": True,
            "tol": 1e-7,
        }
        kmc_args = update_dict(kmc_args, kernel_kwargs)

        if method + "_transition_matrix" not in adata.obsp.keys() or not enforce:
            kmc.fit(
                X,
                V,
                neighbor_idx=indices,
                sample_fraction=sample_fraction,
                **kmc_args,
            )  #

        T = kmc.P
        if correct_density:
            delta_X = kmc.compute_density_corrected_drift(
                X_embedding, kmc.Idx, normalize_vector=True, scale=scale
            )  # indices, k = 500
        else:
            delta_X = kmc.compute_drift(X_embedding, num_prop=1, scale=scale)  # indices, k = 500

        # P = kmc.compute_stationary_distribution()
        # adata.obs['stationary_distribution'] = P

        if calc_rnd_vel:
            kmc = KernelMarkovChain()
            permute_rows_nsign(V)
            kmc.fit(X, V, **kmc_args)  # neighbor_idx=indices,
            T_rnd = kmc.P
            if correct_density:
                delta_X_rnd = kmc.compute_density_corrected_drift(
                    X_embedding, kmc.Idx, normalize_vector=True
                )  # indices, k = 500
            else:
                delta_X_rnd = kmc.compute_drift(X_embedding)
            # P_rnd = kmc.compute_stationary_distribution()
            # adata.obs['stationary_distribution_rnd'] = P_rnd

        adata.uns["kmc"] = kmc
    elif method in ["pearson", "cosine"]:
        vs_kwargs = {
            "n_recurse_neighbors": 2,
            "max_neighs": None,
            "transform": "sqrt",
            "use_neg_vals": True,
        }
        vs_kwargs = update_dict(vs_kwargs, kernel_kwargs)

        if method + "_transition_matrix" in adata.obsp.keys() and not enforce:
            print("Using existing %s found in .obsp." % (method + "_transition_matrix"))
            T = adata.obsp[method + "_transition_matrix"] if transition_matrix is None else transition_matrix
            delta_X = projection_with_transition_matrix(X.shape[0], T, X_embedding, correct_density)
            X_grid, V_grid, D = velocity_on_grid(
                X_embedding[:, :2],
                (X_embedding + delta_X)[:, :2],
                xy_grid_nums=xy_grid_nums,
            )
        else:
            T, delta_X, X_grid, V_grid, D = kernels_from_velocyto_scvelo(
                X,
                X_embedding,
                V,
                indices,
                neg_cells_trick,
                xy_grid_nums,
                method,
                correct_density=correct_density,
                **vs_kwargs,
            )

        if calc_rnd_vel:
            permute_rows_nsign(V)
            (T_rnd, delta_X_rnd, X_grid_rnd, V_grid_rnd, D_rnd,) = kernels_from_velocyto_scvelo(
                X,
                X_embedding,
                V,
                indices,
                neg_cells_trick,
                xy_grid_nums,
                method,
                correct_density=correct_density,
                **vs_kwargs,
            )

    elif method == "fp":
        graph_kwargs = {
            "k": 30,
            "E_func": "sqrt",
            "normalize_v": False,
            "scale_by_dist": False,
        }
        graph_kwargs = update_dict(graph_kwargs, kernel_kwargs)

        fp_kwargs = {"D": 50, "drift_weight": False, "weight_mode": "symmetric"}
        fp_kwargs = update_dict(fp_kwargs, kernel_kwargs)

        wgt_kwargs = {
            "weight": "naive",
            "sig": None,
            "auto_sig_func": None,
            "auto_sig_multiplier": 2,
        }
        wgt_kwargs = update_dict(wgt_kwargs, kernel_kwargs)
        wgt_mode = wgt_kwargs.pop("weight", "naive")

        ctmc_kwargs = {
            "eignum": 30,
        }
        ctmc_kwargs = update_dict(ctmc_kwargs, kernel_kwargs)

        if (
            method + "_transition_matrix" in adata.obsp.keys() or method + "_transition_rate" in adata.obsp.keys()
        ) and not enforce:
            if method + "_transition_matrix" in adata.obsp.keys():
                print("Using existing %s found in .obsp." % (method + "_transition_matrix"))
                T = adata.obsp[method + "_transition_matrix"] if transition_matrix is None else transition_matrix
            elif method + "_transition_rate" in adata.obsp.keys():
                print("Using existing %s found in .obsp." % (method + "_transition_rate"))
                R = adata.obsp[method + "_transition_rate"]
                T = ContinuousTimeMarkovChain(P=R.T).compute_embedded_transition_matrix().T
            delta_X = projection_with_transition_matrix(T.shape[0], T, X_embedding, correct_density)
        else:
            E, nbrs_idx, dists = graphize_velocity(V, X, nbrs_idx=indices, **graph_kwargs)
            if wgt_mode == "naive":
                W = None
            elif wgt_mode == "gaussian":
                main_info("Calculating Gaussian weights with the following parameters:")
                main_info(f"{wgt_kwargs}")
                W = calc_gaussian_weight(nbrs_idx, dists, **wgt_kwargs)
            else:
                raise NotImplementedError(f"The weight mode `{wgt_mode}` is not supported.")

            L = fp_operator(E, W=W, **fp_kwargs)
            ctmc = ContinuousTimeMarkovChain(P=L, **ctmc_kwargs)
            T = sp.csr_matrix(ctmc.compute_embedded_transition_matrix().T)
            delta_X = projection_with_transition_matrix(T.shape[0], T, X_embedding, correct_density)

            adata.obsp["fp_transition_rate"] = ctmc.P.T
            adata.obsp["discrete_vector_field"] = E

    elif method == "transform":
        umap_trans, n_pca_components = (
            adata.uns["umap_fit"]["fit"],
            adata.uns["umap_fit"]["n_pca_components"],
        )

        if "pca_fit" not in adata.uns_keys() or type(adata.uns["pca_fit"]) == str:
            CM = adata.X[:, adata.var.use_for_dynamics.values]
            from ..preprocessing.utils import pca_monocle

            adata, pca_fit, X_pca = pca_monocle(adata, CM, n_pca_components, "X", return_all=True)
            adata.uns["pca_fit"] = pca_fit

        X_pca, pca_fit = adata.obsm["X"], adata.uns["pca_fit"]
        V = adata[:, adata.var.use_for_dynamics.values].layers[vkey] if vkey in adata.layers.keys() else None
        CM, V = CM.A if sp.issparse(CM) else CM, V.A if sp.issparse(V) else V
        V[np.isnan(V)] = 0
        Y_pca = pca_fit.transform(CM + V)

        Y = umap_trans.transform(Y_pca)

        delta_X = Y - X_embedding

    if method not in ["pearson", "cosine"]:
        X_grid, V_grid, D = velocity_on_grid(X_embedding[:, :2], delta_X[:, :2], xy_grid_nums=xy_grid_nums)
        if calc_rnd_vel:
            X_grid_rnd, V_grid_rnd, D_rnd = velocity_on_grid(
                X_embedding[:, :2], delta_X_rnd[:, :2], xy_grid_nums=xy_grid_nums
            )

    if preserve_len:
        basis_len, high_len = np.linalg.norm(delta_X, axis=1), np.linalg.norm(V, axis=1)
        scaler = np.nanmedian(basis_len) / np.nanmedian(high_len)
        for i in LoggerManager.progress_logger(range(adata.n_obs), progress_name="rescaling velocity norm"):
            idx = T[i].indices
            high_len_ = high_len[idx]
            T_i = T[i].data
            delta_X[i] *= T_i.dot(high_len_) / basis_len[i] * scaler

    if add_transition_key is None:
        transition_key = method + "_transition_matrix"
    else:
        transition_key = add_transition_key

    adata.obsp[transition_key] = T
    if add_velocity_key is None:
        velocity_key, grid_velocity_key = "velocity_" + basis, "grid_velocity_" + basis
    else:
        velocity_key, grid_velocity_key = add_velocity_key, "grid_" + add_velocity_key

    adata.obsm[velocity_key] = delta_X
    adata.uns[grid_velocity_key] = {
        "X_grid": X_grid,
        "V_grid": V_grid,
        "D": D,
    }

    if calc_rnd_vel:
        if add_transition_key is None:
            transition_rnd_key = method + "_transition_matrix_rnd"
        else:
            transition_rnd_key = add_transition_key + "_rnd"

        if add_velocity_key is None:
            velocity_rnd_key, grid_velocity_rnd_key = "velocity_" + basis + "_rnd", "grid_velocity_" + basis + "_rnd"
        else:
            velocity_rnd_key, grid_velocity_rnd_key = add_velocity_key + "_rnd", "grid_" + add_velocity_key + "_rnd"

        X_embedding_rnd = "X_" + basis + "_rnd"
        adata.obsp[transition_rnd_key] = T_rnd
        adata.obsm[X_embedding_rnd] = X_embedding
        adata.obsm[velocity_rnd_key] = delta_X_rnd
        adata.uns[grid_velocity_rnd_key] = {
            "X_grid": X_grid_rnd,
            "V_grid": V_grid_rnd,
            "D": D_rnd,
        }

    return adata


def confident_cell_velocities(
    adata,
    group,
    lineage_dict,
    ekey="M_s",
    vkey="velocity_S",
    basis="umap",
    confidence_threshold=0.85,
    only_transition_genes=False,
):
    """Confidently compute transition probability and project high dimension velocity vector to existing low dimension
    embeddings using progenitors and mature cell groups priors.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        group: str
            The column key/name that identifies the cell state grouping information of cells. This will be used for
            calculating gene-wise confidence score in each cell state.
        lineage_dict: dict
            A dictionary describes lineage priors. Keys correspond to the group name from `group` that corresponding
            to the state of one progenitor type while values correspond to the group names from `group` of one or
            multiple terminal cell states. The best practice for determining terminal cell states are those fully
            functional cells instead of intermediate cell states. Note that in python a dictionary key cannot be a list,
            so if you have two progenitor types converge into one terminal cell state, you need to create two records
            each with the same terminal cell as value but different progenitor as the key. Value can be either a string
            for one cell group or a list of string for multiple cell groups.
        ekey: str or None (default: `M_s`)
            The layer that will be used to retrieve data for identifying the gene is in induction or repression phase at
            each cell state. If `None`, .X is used.
        vkey: str or None (default: `velocity_S`)
            The layer that will be used to retrieve velocity data for calculating gene-wise confidence. If `None`,
            `velocity_S` is used.
        basis: str (optional, default `umap`)
            The dictionary key that corresponds to the reduced dimension in `.obsm` attribute.
        confidence_threshold: float (optional, default 0.85)
            The minimal threshold of the mean of the average progenitors and the average mature cells prior based
            gene-wise velocity confidence score. Only genes with score larger than this will be considered as confident
            transition genes for velocity projection.
        only_transition_genes: bool (optional, default False)
            Whether only use previous identified transition genes for confident gene selection, followed by velocity
            projection.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            Returns an updated `~anndata.AnnData` with only confident genes based transition_matrix and projected
            embedding of high dimension velocity vectors in the existing embeddings of current cell state, calculated
            using either the cosine kernel method from (La Manno et al. 2018) or the Itô kernel for the FP method, etc.
    """

    if not any([i.startswith("velocity") for i in adata.layers.keys()]):
        raise Exception(
            "You need to first run `dyn.tl.dynamics(adata)` to estimate kinetic parameters and obtain "
            "raw RNA velocity before running this function."
        )

    if only_transition_genes:
        if "use_for_transition" not in adata.var.keys():
            main_warning(
                "`dyn.tl.cell_velocities(adata)` is not performed yet. Rolling back to use all feature genes "
                "as input for supervised RNA velocity analysis."
            )
            genes = adata.var_names[adata.var.use_for_dynamics]
        else:
            genes = adata.var_names[adata.var.use_for_transition]
    else:
        genes = adata.var_names[adata.var.use_for_dynamics]

    gene_wise_confidence(
        adata,
        group,
        lineage_dict,
        genes=genes,
        ekey=ekey,
        vkey=vkey,
    )

    adata.var.loc[:, "avg_confidence"] = (
        adata.var.loc[:, "avg_prog_confidence"] + adata.var.loc[:, "avg_mature_confidence"]
    ) / 2
    confident_genes = genes[adata[:, genes].var["avg_confidence"] > confidence_threshold]
    adata.var["confident_genes"] = False
    adata.var.loc[confident_genes, "confident_genes"] = True

    X = adata[:, confident_genes].layers[ekey]
    V = adata[:, confident_genes].layers[vkey]
    X_embedding = adata.obsm["X_" + basis]

    cell_velocities(
        adata,
        enforce=True,
        X=X,
        V=V,
        X_embedding=X_embedding,
        basis=basis,
        transition_genes=confident_genes,
    )

    return adata


def stationary_distribution(adata, method="kmc", direction="both", calc_rnd=True):
    """Compute stationary distribution of cells using the transition matrix.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        method: str (default: `kmc`)
            The method to calculate the stationary distribution.
        direction: str (default: `both`)
            The direction of diffusion for calculating the stationary distribution, can be one of `both`, `forward`,
            `backward`.
        calc_rnd: bool (default: True)
            Whether to also calculate the stationary distribution from the control randomized transition matrix.
    Returns
    -------
        adata: :class:`~anndata.AnnData`
            Returns an updated `~anndata.AnnData` with source, sink stationary distributions and the randomized results,
            depending on the direction and calc_rnd arguments.
    """

    # row is the source and columns are targets
    T = adata.obsp["transition_matrix"]

    if method == "kmc":
        kmc = KernelMarkovChain()
        kmc.P = T
        if direction == "both":
            adata.obs["sink_steady_state_distribution"] = kmc.compute_stationary_distribution()
            kmc.P = T.T / T.T.sum(0)
            adata.obs["source_steady_state_distribution"] = kmc.compute_stationary_distribution()

            if calc_rnd:
                T_rnd = adata.obsp["transition_matrix_rnd"]
                kmc.P = T_rnd
                adata.obs["sink_steady_state_distribution_rnd"] = kmc.compute_stationary_distribution()
                kmc.P = T_rnd.T / T_rnd.T.sum(0)
                adata.obs["source_steady_state_distribution_rnd"] = kmc.compute_stationary_distribution()

        elif direction == "forward":
            adata.obs["sink_steady_state_distribution"] = kmc.compute_stationary_distribution()

            if calc_rnd:
                T_rnd = adata.obsp["transition_matrix_rnd"]
                kmc.P = T_rnd
                adata.obs["sink_steady_state_distribution_rnd"] = kmc.compute_stationary_distribution()
        elif direction == "backward":
            kmc.P = T.T / T.T.sum(0)
            adata.obs["source_steady_state_distribution"] = kmc.compute_stationary_distribution()

            if calc_rnd:
                T_rnd = adata.obsp["transition_matrix_rnd"]
                kmc.P = T_rnd.T / T_rnd.T.sum(0)
                adata.obs["sink_steady_state_distribution_rnd"] = kmc.compute_stationary_distribution()

    else:
        T = T.T
        if direction == "both":
            adata.obs["source_steady_state_distribution"] = diffusion(T, backward=True)
            adata.obs["sink_steady_state_distribution"] = diffusion(T)
            if calc_rnd:
                T_rnd = adata.obsp["transition_matrix_rnd"]
                adata.obs["source_steady_state_distribution_rnd"] = diffusion(T_rnd, backward=True)
                adata.obs["sink_steady_state_distribution_rnd"] = diffusion(T_rnd)
        elif direction == "forward":
            adata.obs["sink_steady_state_distribution"] = diffusion(T)
            if calc_rnd:
                T_rnd = adata.uns["transition_matrix_rnd"]
                adata.obs["sink_steady_state_distribution_rnd"] = diffusion(T_rnd)
        elif direction == "backward":
            adata.obs["source_steady_state_distribution"] = diffusion(T, backward=True)
            if calc_rnd:
                T_rnd = adata.obsp["transition_matrix_rnd"]
                adata.obs["source_steady_state_distribution_rnd"] = diffusion(T_rnd, backward=True)


def generalized_diffusion_map(adata, **kwargs):
    """Apply the diffusion map algorithm on the transition matrix build from Itô kernel.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the constructed transition matrix.
        kwargs:
            Additional parameters that will be passed to the diffusion_map_embedding function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that updated with X_diffusion_map embedding in obsm attribute.
    """

    kmc = KernelMarkovChain()
    kmc.P = adata.obsp["transition_matrix"]
    dm_args = {"n_dims": 2, "t": 1}
    dm_args.update(kwargs)
    dm = kmc.diffusion_map_embedding(*dm_args)

    adata.obsm["X_diffusion_map"] = dm


def diffusion(M, P0=None, steps=None, backward=False):
    """Find the state distribution of a Markov process.

    Parameters
    ----------
        M: :class:`~numpy.ndarray` (dimension n x n, where n is the cell number)
            The transition matrix.
        P0: :class:`~numpy.ndarray` (default None; dimension is n, )
            The initial cell state.
        steps: int (default None)
            The random walk steps on the Markov transitioin matrix.
        backward: bool (default False)
            Whether the backward transition will be considered.

    Returns
    -------
        Mu: :class:`~numpy.ndarray`
            The state distribution of the Markov process.
    """

    if backward is True:
        M = M.T
        M = M / M.sum(1)

    if steps is None:
        # code inspired from  https://github.com/prob140/prob140/blob/master/prob140/markov_chains.py#L284

        eigenvalue, eigen = scp.linalg.eig(
            M, left=True, right=False
        )  # if not sp.issparse(M) else eigs(M) # source is on the row

        eigenvector = np.real(eigen) if not sp.issparse(M) else np.real(eigen.T)
        eigenvalue_1_ind = np.isclose(eigenvalue, 1)
        mu = eigenvector[:, eigenvalue_1_ind] / np.sum(eigenvector[:, eigenvalue_1_ind])

        # Zero out floating poing errors that are negative.
        indices = np.logical_and(np.isclose(mu, 0), mu < 0)
        mu[indices] = 0  # steady state distribution

    else:
        mu = (
            np.nanmean(M.dot(np.linalg.matrix_power(M, steps)), 0)
            if P0 is None
            else P0.dot(np.linalg.matrix_power(M, steps))
        )

    return mu


def expected_return_time(M, backward=False):
    """Find the expected returning time.

    Parameters
    ----------
        M: :class:`~numpy.ndarray` (dimension n x n, where n is the cell number)
            The transition matrix.
        backward: bool (default False)
            Whether the backward transition will be considered.

    Returns
    -------
        T: :class:`~numpy.ndarray`
            The expected return time (1 / steady_state_probability).

    """
    steady_state = diffusion(M, P0=None, steps=None, backward=backward)

    T = 1 / steady_state
    return T


def kernels_from_velocyto_scvelo(
    X,
    X_embedding,
    V,
    adj_mat,
    neg_cells_trick,
    xy_grid_nums,
    kernel="pearson",
    n_recurse_neighbors=2,
    max_neighs=None,
    transform="sqrt",
    use_neg_vals=True,
    correct_density=True,
):
    """utility function for calculating the transition matrix and low dimensional velocity embedding via the original
    pearson correlation kernel (La Manno et al., 2018) or the cosine kernel from scVelo (Bergen et al., 2019)."""
    n = X.shape[0]
    if adj_mat is not None:
        rows = []
        cols = []
        vals = []

    delta_X = np.zeros((n, X_embedding.shape[1]))
    for cur_i in LoggerManager.progress_logger(
        range(n),
        progress_name=f"calculating transition matrix via {kernel} kernel with {transform} transform.",
    ):
        velocity = V[cur_i, :]  # project V to pca space

        if velocity.sum() != 0:
            neighbor_index_vals = get_neighbor_indices(
                adj_mat, cur_i, n_recurse_neighbors, max_neighs
            )  # np.zeros((knn, 1))
            diff = X[neighbor_index_vals, :] - X[cur_i, :]

            if transform == "log":
                diff_velocity = np.sign(velocity) * np.log1p(np.abs(velocity))
                diff_rho = np.sign(diff) * np.log1p(np.abs(diff))
            elif transform == "logratio":
                hi_dim, hi_dim_t = X[cur_i, :], X[cur_i, :] + velocity
                log2hidim = np.log1p(np.abs(hi_dim))
                diff_velocity = np.log1p(np.abs(hi_dim_t)) - log2hidim
                diff_rho = np.log1p(np.abs(X[neighbor_index_vals, :])) - np.log1p(np.abs(hi_dim))
            elif transform == "linear":
                diff_velocity = velocity
                diff_rho = diff
            elif transform == "sqrt":
                diff_velocity = np.sign(velocity) * np.sqrt(np.abs(velocity))
                diff_rho = np.sign(diff) * np.sqrt(np.abs(diff))

            if kernel == "pearson":
                vals_ = einsum_correlation(diff_rho, diff_velocity, type="pearson")
            elif kernel == "cosine":
                vals_ = einsum_correlation(diff_rho, diff_velocity, type="cosine")
            rows.extend([cur_i] * len(neighbor_index_vals))
            cols.extend(neighbor_index_vals)
            vals.extend(vals_)
    vals = np.hstack(vals)
    vals[np.isnan(vals)] = 0
    G = sp.csr_matrix((vals, (rows, cols)), shape=(X_embedding.shape[0], X_embedding.shape[0]))
    G = split_velocity_graph(G, neg_cells_trick)

    if neg_cells_trick:
        G, G_ = G

    confidence, ub_confidence = G.max(1).A.flatten(), np.percentile(G.max(1).A.flatten(), 98)
    dig_p = np.clip(ub_confidence - confidence, 0, 1)
    G.setdiag(dig_p)

    T = np.expm1(G / 0.1)

    if neg_cells_trick:
        if use_neg_vals:
            T -= np.expm1(-G_ / 0.1)
        else:
            T += np.expm1(G_ / 0.1)
            T.data = T.data + 1

    # T = w * (~ direct_neighs).multiply(T) + (1 - w) * direct_neighs.multiply(T)

    # normalize so that each row sum up to 1
    sparsefuncs.inplace_row_scale(T, 1 / np.abs(T).sum(axis=1).A1)

    T.setdiag(0)
    T.eliminate_zeros()

    delta_X = projection_with_transition_matrix(n, T, X_embedding, correct_density)

    X_grid, V_grid, D = velocity_on_grid(
        X_embedding[:, :2],
        (X_embedding + delta_X)[:, :2],
        xy_grid_nums=xy_grid_nums,
    )

    return T, delta_X, X_grid, V_grid, D


def projection_with_transition_matrix(n, T, X_embedding, correct_density=True):
    delta_X = np.zeros((n, X_embedding.shape[1]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in LoggerManager.progress_logger(
            range(n),
            progress_name="projecting velocity vector to low dimensional embedding",
        ):
            idx = T[i].indices
            diff_emb = X_embedding[idx] - X_embedding[i, None]
            diff_emb /= norm(diff_emb, axis=1)[:, None]
            if np.isnan(diff_emb).sum() != 0:
                diff_emb[np.isnan(diff_emb)] = 0
            T_i = T[i].data
            delta_X[i] = T_i.dot(diff_emb)
            if correct_density:
                delta_X[i] -= T_i.mean() * diff_emb.sum(0)

    return delta_X


# utility functions for calculating the random cell velocities
@jit(nopython=True)
def numba_random_seed(seed):
    """Same as np.random.seed but for numba. Function adapated from velocyto.

    Parameters
    ----------
        seed: int
            Random seed value

    """
    np.random.seed(seed)


@jit(nopython=True)
def permute_rows_nsign(A):
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently. Function
    adapted from velocyto

    Parameters
    ----------
        A: :class:`~numpy.ndarray`
            A numpy array that will be permuted.
    """

    plmi = np.array([+1, -1])
    for i in range(A.shape[1]):
        np.random.shuffle(A[:, i])
        A[:, i] = A[:, i] * np.random.choice(plmi, size=A.shape[0])


"""This function can be removed now
def embed_velocity(
    adata,
    x_basis,
    v_basis="velocity",
    emb_basis="X",
    velocity_gene_tag="transition_genes",
    num_pca=100,
    n_recurse_neighbors=2,
    M_diff=0.25,
    adaptive_local_kernel=True,
    normalize_velocity=True,
    return_kmc=False,
    **kmc_kwargs,
):
    if velocity_gene_tag is not None:
        X = adata.layers[x_basis][:, adata.var[velocity_gene_tag]]
        V = adata.layers[v_basis][:, adata.var[velocity_gene_tag]]
    else:
        X = adata.layers[x_basis]
        V = adata.layers[v_basis]

    X = log1p_(adata, X)

    X_emb = adata.obsm[emb_basis]
    Idx = adata.uns["neighbors"]["indices"]

    if num_pca is not None:
        pca = PCA()
        pca.fit(X)
        X_pca = pca.transform(X)
        Y_pca = pca.transform(X + V)
        V_pca = Y_pca - X_pca
    else:
        X_pca = X
        V_pca = V

    kmc = KernelMarkovChain()
    kmc.fit(
        X_pca[:, :num_pca],
        V_pca[:, :num_pca],
        neighbor_idx=Idx,
        n_recurse_neighbors=n_recurse_neighbors,
        M_diff=M_diff,
        adaptive_local_kernel=adaptive_local_kernel,
        **kmc_kwargs,
    )

    Uc = kmc.compute_density_corrected_drift(X_emb, normalize_vector=normalize_velocity)
    if return_kmc:
        return Uc, kmc
    else:
        return Uc"""
