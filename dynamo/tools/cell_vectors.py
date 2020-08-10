from tqdm import tqdm
import warnings
import scipy as scp
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import PCA
from sklearn.utils import sparsefuncs
from .Markov import *
from .connectivity import extract_indices_dist_from_graph
from .topography import VectorField
from .vector_calculus import acceleration
from .utils import (
    set_velocity_genes,
    get_finite_inds,
    get_ekey_vkey_from_adata,
    get_mapper_inverse,
    update_dict,
    get_iterative_indices,
    split_velocity_graph,
    norm_row,
    einsum_correlation,
    build_distance_graph,
    log1p_,
)

from .dimension_reduction import reduceDimension

def cell_velocities(
    adata,
    ekey=None,
    vkey=None,
    X=None,
    V_mat=None,
    X_embedding=None,
    use_mnn=False,
    neighbors_from_basis=False,
    n_pca_components=None,
    min_r2=0.01,
    min_alpha=0.01,
    min_gamma=0.01,
    min_delta=0.01,
    basis="umap",
    method="pearson",
    neg_cells_trick=True,
    calc_rnd_vel=False,
    xy_grid_nums=(50, 50),
    correct_density=True,
    scale=True,
    sample_fraction=None,
    random_seed=19491001,
    other_kernels_dict={},
    enforce=False,
    key=None,
    preserve_len=False,
    **kmc_kwargs
):
    """Compute transition probability and project high dimension velocity vector to existing low dimension embedding.

    It is powered by the Itô kernel that not only considers the correlation between the vector from any cell to its
    nearest neighbors and its velocity vector but also the corresponding distances. We expect this new kernel will enable
    us to visualize more intricate vector flow or steady states in low dimension. We also expect it will improve the
    calculation of the stationary distribution or source states of sampled cells. The original "correlation/cosine"
    velocity projection method is also supported. Kernels based on the reconstructed velocity field is also possible.

    With the `key` argument, `cell_velocities` can be called by `cell_accelerations` to calculate RNA acceleration vector
    for each cell.

    Arguments
    ---------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        ekey: `str` or None (optional, default `None`)
            The dictionary key that corresponds to the gene expression in the layer attribute. By default, ekey and vkey
            will be automatically detected from the adata object.
        vkey: 'str' or None (optional, default `None`)
            The dictionary key that corresponds to the estimated velocity values in the layers attribute.
        X: 'np.ndarray' or `sp.csr_matrix` or None (optional, default `None`)
            The expression states of single cells (or expression states in reduced dimension, like pca, of single cells)
        V_mat: 'np.ndarray' or `sp.csr_matrix` or None (optional, default `None`)
            The RNA velocity of single cells (or velocity estimates projected to reduced dimension, like pca, of single
            cells). Note that X, V_mat need to have the exact dimensionalities.
        X_embedding: 'str' or None (optional, default `None`)
            The low expression reduced space (pca, umap, tsne, etc.) of single cells that RNA velocity will be projected
            onto. Note X_embedding, X and V_mat has to have the same cell/sample dimension and X_embedding should have
            less feature dimension comparing that of X or V_mat.
        use_mnn: `bool` (optional, default `False`)
            Whether to use mutual nearest neighbors for projecting the high dimensional velocity vectors. By default, we
            don't use the mutual nearest neighbors. Mutual nearest neighbors are calculated from nearest neighbors across
            different layers, which which accounts for cases where, for example, the cells from spliced expression may be
            nearest neighbors but far from nearest neighbors on unspliced data. Using mnn assumes your data from different
            layers are reliable (otherwise it will destroy real signals).
        neighbors_from_basis: `bool` (optional, default `False`)
            Whether to construct nearest neighbors from low dimensional space as defined by the `basis`, instead of using
            that calculated during UMAP process.
        n_pca_components: `int` (optional, default `None`)
            The number of pca components to project the high dimensional X, V before calculating transition matrix for
            velocity visualization. By default it is None and if method is `kmc`, n_pca_components will be reset to 30;
            otherwise use all high dimensional data for velocity projection.
        min_r2: `float` (optional, default `0.01`)
            The minimal value of r-squared of the parameter fits for selecting velocity genes.
        min_alpha: `float` (optional, default `0.01`)
            The minimal value of alpha kinetic parameter for selecting velocity genes.
        min_gamma: `float` (optional, default `0.01`)
            The minimal value of gamma kinetic parameter for selecting velocity genes.
        min_delta: `float` (optional, default `0.01`)
            The minimal value of delta kinetic parameter for selecting velocity genes.
        basis: 'int' (optional, default `umap`)
            The dictionary key that corresponds to the reduced dimension in `.obsm` attribute.
        method: `string` (optional, default `pearson`)
            The method to calculate the transition matrix and project high dimensional vector to low dimension, either `kmc`,
            `cosine`, `pearson`, or `transform`. "kmc" is our new approach to learn the transition matrix via diffusion
            approximation or an Itô kernel. "cosine" or "pearson" are the methods used in the original RNA velocity paper
            or the scvelo paper (Note that scVelo implementation actually centers both dX and V, so its cosine kernel is
            equivalent to pearson correlation kernel but we also provide the raw cosine kernel). "kmc" option is arguable
            better than "correlation" or "cosine" as it not only considers the correlation but also the distance of the
            nearest neighbors to the high dimensional velocity vector. Finally, the "transform" method uses umap's transform
            method to transform new data points to the UMAP space. "transform" method is NOT recommended. Kernels that
            are based on the reconstructed vector field in high dimension is also possible.
        neg_cells_trick: 'bool' (optional, default `True`)
            Whether we should handle cells having negative correlations in gene expression difference with high dimensional
            velocity vector separately. This option was borrowed from scVelo package (https://github.com/theislab/scvelo)
            and use in conjunction with "pearson" and "cosine" kernel. Not required if method is set to be "kmc".
        calc_rnd_vel: `bool` (default: `False`)
            A logic flag to determine whether we will calculate the random velocity vectors which can be plotted
            downstream as a negative control and used to adjust the quiver scale of the velocity field.
        xy_grid_nums: `tuple` (default: `(50, 50)`).
            A tuple of number of grids on each dimension.
        correct_density: `bool` (default: `False`)
            Whether to correct density when calculating the markov transition matrix, applicable to the `kmc` kernel.
        correct_density: `bool` (default: `False`)
            Whether to scale velocity when calculating the markov transition matrix, applicable to the `kmc` kernel.
        sample_fraction: `None` or `float` (default: `None`)
            The downsampled fraction of kNN for the purpose of acceleration, applicable to the `kmc` kernel.
        random_seed: `int` (default: `19491001`)
            The random seed for numba to ensure consistency of the random velocity vectors. Default value 19491001 is a
            special day for those who care.
        key: `str` or None (default: `None`)
            The prefix key that will be prefixed to the keys for storing calculated transition matrix, projection vectors, etc.
        preserve_len: `bool` (default: `False`)
            Whether to preserve the length of high dimension vector length. When set to be True, the length  of low
            dimension projected vector will be proportionally scaled to that of the high dimensional vector.
        other_kernels_dict: `dict` (default: `{}`)
            A dictionary of paramters that will be passed to the cosine/correlation kernel.
        enforce: `bool` (default: `False`)
            Whether to enforce 1) redefining use_for_velocity column in obs attribute;
                               2) recalculation of transition matrix.

    Returns
    -------
        Adata: :class:`~anndata.AnnData`
            Returns an updated `~anndata.AnnData` with transition_matrix and projected embedding of high dimension velocity
            vectors in the existing embeddings of current cell state, calculated using either the Itô kernel method
            (default) or the diffusion approximation or the method from (La Manno et al. 2018).
    """

    mapper_r = get_mapper_inverse()
    layer = mapper_r[ekey] if (ekey is not None and ekey in mapper_r.keys()) else ekey
    ekey, vkey, layer = (
        get_ekey_vkey_from_adata(adata)
        if (ekey is None or vkey is None)
        else (ekey, vkey, layer)
    )

    if calc_rnd_vel:
        numba_random_seed(random_seed)

    if (not neighbors_from_basis) and ("neighbors" in adata.uns.keys()):
        if use_mnn:
            neighbors = adata.uns["mnn"]
            indices, dist = extract_indices_dist_from_graph(
                neighbors, adata.uns["neighbors"]["indices"].shape[1]
            )
            indices, dist = indices[:, 1:], dist[:, 1:]
        else:
            if adata.obsp["distances"].shape[0] == adata.obsp["distances"].shape[1]:
                knn_indices, knn_dists = extract_indices_dist_from_graph(
                    adata.obsp["distances"], 30
                    # np.min((adata.uns["neighbors"]["connectivities"] > 0).sum(1).A)
                )
                knn_dists = build_distance_graph(knn_indices, knn_dists)

                adata.uns["neighbors"]["indices"], adata.obsp["distances"] = knn_indices, knn_dists
            neighbors, dist, indices = (
                adata.obsp["connectivities"],
                adata.obsp["distances"],
                adata.uns["neighbors"]["indices"],
            )
            indices, dist = indices[:, 1:], dist[:, 1:]


    if 'use_for_velocity' not in adata.var.keys() or enforce:
        use_for_dynamics = True if "use_for_dynamics" in adata.var.keys() else False
        adata = set_velocity_genes(
            adata, vkey="velocity_S", min_r2=min_r2, use_for_dynamics=use_for_dynamics,
            min_alpha=min_alpha, min_gamma=min_gamma, min_delta=min_delta,
        )

    X = adata[:, adata.var.use_for_velocity.values].layers[ekey] if X is None else X
    V_mat = (
        adata[:, adata.var.use_for_velocity.values].layers[vkey]
        if vkey in adata.layers.keys()
        else None
    ) if V_mat is None else V_mat

    if X.shape != V_mat.shape and X.shape[0] != adata.n_obs:
        raise Exception(f"X and V_mat doesn't have the same dimensionalities or X/V_mat doesn't {adata.n_obs} rows!")
    
    if X_embedding is None:
        if vkey == "velocity_S":
            X_embedding = adata.obsm["X_" + basis]
        else:
            adata = reduceDimension(adata, layer=layer, reduction_method=basis)
            X_embedding = adata.obsm[layer + "_" + basis]

    if X.shape[0] != X_embedding.shape[0] and X.shape[1] > X_embedding.shape[1]:
        raise Exception(f"X and X_embedding doesn't have the same sample dimension or "
                        f"X doesn't have the higher feature dimension!")

    V_mat = V_mat.A if issparse(V_mat) else V_mat
    X = X.A if issparse(X) else X
    finite_inds = get_finite_inds(V_mat)
    X, V_mat = X[:, finite_inds], V_mat[:, finite_inds]

    if method == 'kmc' and n_pca_components is None: n_pca_components = 30
    if n_pca_components is not None:
        X = log1p_(adata, X)
        X_plus_V = log1p_(adata, X + V_mat)
        if (
                "velocity_pca_fit" not in adata.uns_keys()
                or type(adata.uns["velocity_pca_fit"]) == str
        ):
            pca = PCA(
                n_components=min(n_pca_components, X.shape[1] - 1),
                svd_solver="arpack",
                random_state=0,
            )
            pca_fit = pca.fit(X)
            X_pca = pca_fit.transform(X)

            adata.uns["velocity_pca_fit"] = pca_fit
            adata.uns["velocity_PCs"] = pca_fit.components_.T
            adata.obsm["X_velocity_pca"] = X_pca

        X_pca, PCs, pca_fit = (
            adata.obsm["X_velocity_pca"],
            adata.uns["velocity_PCs"],
            adata.uns["velocity_pca_fit"],
        )

        Y_pca = pca_fit.transform(X_plus_V)
        V_pca = Y_pca - X_pca
        # V_pca = (V_mat - V_mat.mean(0)).dot(PCs)

        adata.obsm["velocity_pca_raw"] = V_pca
        X, V_mat = X_pca[:, :n_pca_components], V_pca[:, :n_pca_components]

    if neighbors_from_basis:
        if X.shape[0] > 200000 and X.shape[1] > 2: 
            from pynndescent import NNDescent

            nbrs = NNDescent(X, metric='eulcidean', n_neighbors=30, n_jobs=-1,
                              random_state=19490110, **kwargs)
            indices, _ = nbrs.query(X, k=30)
        else:
            alg = "ball_tree" if X.shape[1] > 10 else 'kd_tree'
            nbrs = NearestNeighbors(n_neighbors=30, algorithm=alg, n_jobs=-1).fit(X)
            _, indices = nbrs.kneighbors(X)

    # add both source and sink distribution
    if method == "kmc":
        if method + '_transition_matrix' in adata.uns_keys() and not enforce:
            T = adata.uns[method + '_transition_matrix']
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
        kmc_args = update_dict(kmc_args, kmc_kwargs)

        if method + '_transition_matrix' not in adata.uns_keys() or not enforce:
            kmc.fit(
                X,
                V_mat,
                neighbor_idx=indices,
                sample_fraction=sample_fraction,
                **kmc_args
            )  #

        T = kmc.P
        if correct_density:
            delta_X = kmc.compute_density_corrected_drift(
                X_embedding, kmc.Idx, normalize_vector=True, scale=scale
            )  # indices, k = 500
        else:
            delta_X = kmc.compute_drift(
                X_embedding, num_prop=1, scale=scale
            )  # indices, k = 500

        # P = kmc.compute_stationary_distribution()
        # adata.obs['stationary_distribution'] = P
        X_grid, V_grid, D = velocity_on_grid(
            X_embedding, delta_X, xy_grid_nums=xy_grid_nums
        )

        if calc_rnd_vel:
            kmc = KernelMarkovChain()
            permute_rows_nsign(V_mat)
            kmc.fit(X, V_mat, **kmc_args)  # neighbor_idx=indices,
            T_rnd = kmc.P
            if correct_density:
                delta_X_rnd = kmc.compute_density_corrected_drift(
                    X_embedding, kmc.Idx, normalize_vector=True
                )  # indices, k = 500
            else:
                delta_X_rnd = kmc.compute_drift(X_embedding)
            # P_rnd = kmc.compute_stationary_distribution()
            # adata.obs['stationary_distribution_rnd'] = P_rnd
            X_grid_rnd, V_grid_rnd, D_rnd = velocity_on_grid(
                X_embedding, delta_X_rnd, xy_grid_nums=xy_grid_nums
            )

        adata.uns["kmc"] = kmc
    elif method in ["pearson", "cosine"]:
        vs_kwargs = {"n_recurse_neighbors": 2,
                      "max_neighs": None,
                      "transform": 'sqrt',
                      "use_neg_vals": True,
                     }
        vs_kwargs = update_dict(vs_kwargs, other_kernels_dict)

        if method + '_transition_matrix' in adata.uns_keys() and not enforce:
            T = adata.uns[method + '_transition_matrix']
            delta_X = projection_with_transition_matrix(X.shape[0], T, X_embedding)
            X_grid, V_grid, D = velocity_on_grid(
                X_embedding[:, :2], (X_embedding + delta_X)[:, :2], xy_grid_nums=xy_grid_nums
            )
        else:
            T, delta_X, X_grid, V_grid, D = kernels_from_velocyto_scvelo(
                X, X_embedding, V_mat, indices, neg_cells_trick, xy_grid_nums, neighbors,
                method, **vs_kwargs
            )

        if calc_rnd_vel:
            permute_rows_nsign(V_mat)
            T_rnd, delta_X_rnd, X_grid_rnd, V_grid_rnd, D_rnd = kernels_from_velocyto_scvelo(
                X, X_embedding, V_mat, indices, neg_cells_trick, xy_grid_nums, neighbors,
                method, **vs_kwargs
            )
    elif method == "transform":
        umap_trans, n_pca_components = (
            adata.uns["umap_fit"]["fit"],
            adata.uns["umap_fit"]["n_pca_components"],
        )

        if "pca_fit" not in adata.uns_keys() or type(adata.uns["pca_fit"]) == str:
            CM = adata.X[:, adata.var.use_for_dynamics.values]
            from ..preprocessing.utils import pca

            adata, pca_fit, X_pca = pca(adata, CM, n_pca_components, "X")
            adata.uns["pca_fit"] = pca_fit

        X_pca, pca_fit = adata.obsm["X"], adata.uns["pca_fit"]
        V = (
            adata[:, adata.var.use_for_dynamics.values].layers[vkey]
            if vkey in adata.layers.keys()
            else None
        )
        CM, V = CM.A if issparse(CM) else CM, V.A if issparse(V) else V
        V[np.isnan(V)] = 0
        Y_pca = pca_fit.transform(CM + V)

        Y = umap_trans.transform(Y_pca)

        delta_X = Y - X_embedding

        X_grid, V_grid, D = velocity_on_grid(
            X_embedding, delta_X, xy_grid_nums=xy_grid_nums
        ),

    if preserve_len:
        basis_len, high_len = np.linalg.norm(delta_X, axis=1), np.linalg.norm(V_mat, axis=1)
        scaler = np.nanmedian(basis_len) / np.nanmedian(high_len)
        for i in tqdm(range(adata.n_obs), desc=f"rescaling velocity norm..."):
            idx = T[i].indices
            high_len_ = high_len[idx]
            T_i = T[i].data
            delta_X[i] *= T_i.dot(high_len_) / basis_len[i] * scaler

    if key is None:
        adata.uns[method + "_transition_matrix"] = T
        adata.obsm["velocity_" + basis] = delta_X
        adata.uns["grid_velocity_" + basis] = {"X_grid": X_grid, "V_grid": V_grid, "D": D}
    else:
        adata.uns[key + '_' + method + "_transition_matrix"] = T
        adata.obsm[key + '_' + basis] = delta_X
        adata.uns["grid_" + key + '_' + basis] = {"X_grid": X_grid, "V_grid": V_grid, "D": D}

    if calc_rnd_vel:
        if key is None:
            adata.uns[method + "_transition_matrix_rnd"] = T_rnd
            adata.obsm["X_" + basis + "_rnd"] = X_embedding
            adata.obsm["velocity_" + basis + "_rnd"] = delta_X_rnd
            adata.uns["grid_velocity_" + basis + "_rnd"] = {
                "X_grid": X_grid_rnd,
                "V_grid": V_grid_rnd,
                "D": D_rnd,
            }
        else:
            adata.uns[key + '_' + method + "_transition_matrix_rnd"] = T_rnd
            adata.obsm["X_" + key + "_" + basis + "_rnd"] = X_embedding
            adata.obsm[key + "_" + basis + "_rnd"] = delta_X_rnd
            adata.uns["grid_" + key + '_' + basis + "_rnd"] = {
                "X_grid": X_grid_rnd,
                "V_grid": V_grid_rnd,
                "D": D_rnd,
            }

    return adata


def cell_accelerations(adata,
                       vf_basis='pca',
                       basis='umap',
                       enforce=True,
                       preserve_len=True,
                       other_kernels_dict={},
                       **kwargs):
    """Compute transition probability and project high dimension acceleration vector to existing low dimension embedding.

    In classical physics, including fluidics and aerodynamics, velocity and acceleration vector fields are used as
    fundamental tools to describe motion or external force of objects, respectively. In analogy, RNA velocity or
    accelerations estimated from single cells can be regarded as samples in the velocity (La Manno et al. 2018) or
    acceleration vector field (Gorin, Svensson, and Pachter 2019). In general, a vector field can be defined as a
    vector-valued function f that maps any points (or cells’ expression state) x in a domain Ω with D dimension (or the
    gene expression system with D transcripts / proteins) to a vector y (for example, the velocity or acceleration for
    different genes or proteins), that is f(x) = y.

    In two or three dimensions, a velocity vector field is often visualised as a quiver plot where a collection of arrows
    with a given magnitude and direction is drawn. For example, the velocity estimates of unspliced transcriptome of
    sampled cells projected into two dimensions is drawn to show the prediction of the future cell states in RNA velocity
    (La Manno et al. 2018). During the differentiation process, external signal factors perturb cells and thus change
    the vector field. Since we perform genome-wide profiling of cell states and the experiments performed are often done
    in a short time scale, we assume a constant vector field without loss of generality (See also Discussion). Assuming
    an asymptotic deterministic system, the trajectory of the cells travelling in the gene expression space follows the
    vector field and can be calculated using numerical integration methods, for example Runge-Kutta algorithm. In two or
    three dimensions, a streamline plot can be used to visualize the paths of cells will follow if released in different
    regions of the gene expression state space under a steady flow field. Another more intuitive way to visualize the
    structure of vector field is the so called line integral convolution method or LIC (Cabral and Leedom 1993), which
    works by adding random black-and-white paint sources on the vector field and letting the flowing particle on the
    vector field picking up some texture to ensure the same streamline having similar intensity. Although we have not
    provides such functionalities in dynamo, with vector field that changes over time, similar methods, for example,
    streakline, pathline, timeline, etc. can be used to visualize the evolution of single cell or cell populations.

    Arguments
    ---------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        vf_basis: 'int' (optional, default `pca`)
            The dictionary key that corresponds to the low dimensional embedding where the vector field function
            reconstructed.
        basis: 'int' (optional, default `umap`)
            The dictionary key that corresponds to the reduced dimension in `.obsm` attribute.
        enforce: `bool` (default: `False`)
            Whether to enforce 1) redefining use_for_velocity column in obs attribute;
                               2) recalculation of transition matrix.
        preserve_len: `bool` (default: `True`)
            Whether to preserve the length of high dimension vector length. When set to be True, the length  of low
            dimension projected vector will be proportionally scaled to that of the high dimensional vector. Note that
            when preserve_len is set to be True, the acceleration field may seem to be messy (although the magnitude will
            be reflected) while the trend of acceleration when `preserve_len` is `True` is more clearer but will lose
            information of acceleration magnitude. This is because the acceleration is not directly related to the
            distance of cells in the low embedding space; thus the acceleration direction can be better preserved than
            the magnitude. On the other hand, velocity is more relevant to the distance in low embedding space, so
            preserving magnitude and direction of velocity vector in low dimension can be more easily achieved.
        other_kernels_dict: `dict` (default: `{}`)
            A dictionary of paramters that will be passed to the cosine/correlation kernel.

    Returns
    -------
        Adata: :class:`~anndata.AnnData`
            Returns an updated `~anndata.AnnData` with transition_matrix and projected embedding of high dimension velocity
            vectors in the existing embeddings of current cell state, calculated using either the Itô kernel method
            (default) or the diffusion approximation or the method from (La Manno et al. 2018).
    """

    if 'velocity_' + vf_basis not in adata.obsm.keys():
        cell_velocities(adata, basis=vf_basis)

    if 'VecFld_' + vf_basis not in adata.uns_keys():
        VectorField(adata, basis=vf_basis)

    if 'acceleration_' + vf_basis not in adata.obsm.keys():
        acceleration(adata, basis=vf_basis)

    X = adata.obsm['X_' + vf_basis]
    V_mat = adata.obsm['acceleration_' + vf_basis]
    X_embedding = adata.obsm['X_' + basis]

    if basis != vf_basis and vf_basis.lower() not in ['umap', 'tsne', 'trimap', 'ddtree', 'diffusion_map']:
        cell_velocities(
            adata,
            X=X,
            V_mat=V_mat,
            X_embedding=X_embedding,
            basis=basis,
            enforce=enforce,
            key='acceleration',
            preserve_len=preserve_len,
            other_kernels_dict=other_kernels_dict,
            **kwargs
        )


def stationary_distribution(adata, method="kmc", direction="both", calc_rnd=True):
    """Compute stationary distribution of cells using the transition matrix.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        method: `str` (default: `kmc`)
            The method to calculate the stationary distribution.
        direction: `str` (default: `both`)
            The direction of diffusion for calculating the stationary distribution, can be one of `both`, `forward`, `backward`.
        calc_rnd: `bool` (default: `True`)
            Whether to also calculate the stationary distribution from the control randomized transition matrix.
    Returns
    -------
        Adata: :class:`~anndata.AnnData`
            Returns an updated `~anndata.AnnData` with source, sink stationary distributions and the randomized results,
            depending on the direction and calc_rnd arguments.
    """

    T = adata.uns["transition_matrix"]  # row is the source and columns are targets

    if method == "kmc":
        kmc = KernelMarkovChain()
        kmc.P = T
        if direction == "both":
            adata.obs[
                "sink_steady_state_distribution"
            ] = kmc.compute_stationary_distribution()
            kmc.P = T.T / T.T.sum(0)
            adata.obs[
                "source_steady_state_distribution"
            ] = kmc.compute_stationary_distribution()

            if calc_rnd:
                T_rnd = adata.uns["transition_matrix_rnd"]
                kmc.P = T_rnd
                adata.obs[
                    "sink_steady_state_distribution_rnd"
                ] = kmc.compute_stationary_distribution()
                kmc.P = T_rnd.T / T_rnd.T.sum(0)
                adata.obs[
                    "source_steady_state_distribution_rnd"
                ] = kmc.compute_stationary_distribution()

        elif direction == "forward":
            adata.obs[
                "sink_steady_state_distribution"
            ] = kmc.compute_stationary_distribution()

            if calc_rnd:
                T_rnd = adata.uns["transition_matrix_rnd"]
                kmc.P = T_rnd
                adata.obs[
                    "sink_steady_state_distribution_rnd"
                ] = kmc.compute_stationary_distribution()
        elif direction == "backward":
            kmc.P = T.T / T.T.sum(0)
            adata.obs[
                "source_steady_state_distribution"
            ] = kmc.compute_stationary_distribution()

            if calc_rnd:
                T_rnd = adata.uns["transition_matrix_rnd"]
                kmc.P = T_rnd.T / T_rnd.T.sum(0)
                adata.obs[
                    "sink_steady_state_distribution_rnd"
                ] = kmc.compute_stationary_distribution()

    else:
        T = T.T
        if direction == "both":
            adata.obs["source_steady_state_distribution"] = diffusion(T, backward=True)
            adata.obs["sink_steady_state_distribution"] = diffusion(T)
            if calc_rnd:
                T_rnd = adata.uns["transition_matrix_rnd"]
                adata.obs["source_steady_state_distribution_rnd"] = diffusion(
                    T_rnd, backward=True
                )
                adata.obs["sink_steady_state_distribution_rnd"] = diffusion(T_rnd)
        elif direction == "forward":
            adata.obs["sink_steady_state_distribution"] = diffusion(T)
            if calc_rnd:
                T_rnd = adata.uns["transition_matrix_rnd"]
                adata.obs["sink_steady_state_distribution_rnd"] = diffusion(T_rnd)
        elif direction == "backward":
            adata.obs["source_steady_state_distribution"] = diffusion(T, backward=True)
            if calc_rnd:
                T_rnd = adata.uns["transition_matrix_rnd"]
                adata.obs["source_steady_state_distribution_rnd"] = diffusion(
                    T_rnd, backward=True
                )


def generalized_diffusion_map(adata, **kwargs):
    """Apply the diffusion map algorithm on the transition matrix build from Itô kernel.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the constructed transition matrix.'
        kwargs:
            Additional parameters that will be passed to the diffusion_map_embedding function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that updated with X_diffusion_map embedding in obsm attribute.
    """

    kmc = KernelMarkovChain()
    kmc.P = adata.uns["transition_matrix"]
    dm_args = {"n_dims": 2, "t": 1}
    dm_args.update(kwargs)
    dm = kmc.diffusion_map_embedding(*dm_args)

    adata.obsm["X_diffusion_map"] = dm


def diffusion(M, P0=None, steps=None, backward=False):
    """Find the state distribution of a Markov process.

    Parameters
    ----------
        M: `numpy.ndarray` (dimension n x n, where n is the cell number)
            The transition matrix.
        P0: `numpy.ndarray` (default None; dimension is n, )
            The initial cell state.
        steps: `int` (default None)
            The random walk steps on the Markov transitioin matrix.
        backward: `bool` (default False)
            Whether the backward transition will be considered.

    Returns
    -------
        Mu: `numpy.ndarray`
            The state distribution of the Markov process.
    """

    if backward is True:
        M = M.T
        M = M / M.sum(1)

    if steps is None:
        # code inspired from  https://github.com/prob140/prob140/blob/master/prob140/markov_chains.py#L284
        from scipy.sparse.linalg import eigs

        eigenvalue, eigenvector = scp.linalg.eig(
            M, left=True, right=False
        )  # if not issparse(M) else eigs(M) # source is on the row

        eigenvector = (
            np.real(eigenvector) if not issparse(M) else np.real(eigenvector.T)
        )
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
        M: `numpy.ndarray` (dimension n x n, where n is the cell number)
            The transition matrix.
        backward: `bool` (default False)
            Whether the backward transition will be considered.

    Returns
    -------
        T: `numpy.ndarray`
            The expected return time (1 / steady_state_probability).

    """
    steady_state = diffusion(M, P0=None, steps=None, backward=backward)

    T = 1 / steady_state
    return T


def kernels_from_velocyto_scvelo(
    X, X_embedding, V_mat, indices, neg_cells_trick, xy_grid_nums, neighbors,
    kernel='pearson', n_recurse_neighbors=2, max_neighs=None, transform='sqrt',
    use_neg_vals=True,
):
    """utility function for calculating the transition matrix and low dimensional velocity embedding via the original
    pearson correlation kernel (La Manno et al., 2018) or the cosine kernel from scVelo (Bergen et al., 2019)."""
    n = X.shape[0]
    if indices is not None:
        rows = []
        cols = []
        vals = []

    delta_X = np.zeros((n, X_embedding.shape[1]))
    for i in tqdm(range(n), desc=f"calculating transition matrix via {kernel} kernel with {transform} transform."):
        velocity = V_mat[i, :]  # project V_mat to pca space

        if velocity.sum() != 0:
            i_vals = get_iterative_indices(indices, i, n_recurse_neighbors, max_neighs)  # np.zeros((knn, 1))
            diff = X[i_vals, :] - X[i, :]

            if transform == 'log':
                diff_velocity = np.sign(velocity) * np.log(np.abs(velocity) + 1)
                diff_rho = np.sign(diff) * np.log(np.abs(diff) + 1)
            elif transform == 'logratio':
                hi_dim, hi_dim_t = X[i, :], X[i, :] + velocity
                log2hidim = np.log(np.abs(hi_dim) + 1)
                diff_velocity = np.log(np.abs(hi_dim_t) + 1) - log2hidim
                diff_rho = np.log(np.abs(X[i_vals, :]) + 1) - np.log(np.abs(hi_dim) + 1)
            elif transform == 'linear':
                diff_velocity = velocity
                diff_rho = diff
            elif transform == 'sqrt':
                diff_velocity = np.sign(velocity) * np.sqrt(np.abs(velocity))
                diff_rho = np.sign(diff) * np.sqrt(np.abs(diff))

            if kernel == 'pearson':
                vals_ = einsum_correlation(diff_rho, diff_velocity, type="pearson")
            elif kernel == 'cosine':
                vals_ = einsum_correlation(diff_rho, diff_velocity, type="cosine")

            rows.extend([i] * len(i_vals))
            cols.extend(i_vals)
            vals.extend(vals_)

    vals = np.hstack(vals)
    vals[np.isnan(vals)] = 0
    G = csr_matrix(
        (vals, (rows, cols)), shape=neighbors.shape
    )
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

    delta_X = projection_with_transition_matrix(n, T, X_embedding)

    X_grid, V_grid, D = velocity_on_grid(
        X_embedding[:, :2], (X_embedding + delta_X)[:, :2], xy_grid_nums=xy_grid_nums
    )

    return T, delta_X, X_grid, V_grid, D


def projection_with_transition_matrix(n, T, X_embedding):
    delta_X = np.zeros((n, X_embedding.shape[1]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i in tqdm(range(n), desc=f"projecting velocity vector to low dimensional embedding..."):
            idx = T[i].indices
            diff_emb = X_embedding[idx] - X_embedding[i, None]
            diff_emb /= norm_row(diff_emb)[:, None]
            diff_emb[np.isnan(diff_emb)] = 0
            T_i = T[i].data
            delta_X[i] = T_i.dot(diff_emb) - T_i.mean() * diff_emb.sum(0)

    return delta_X


# utility functions for calculating the random cell velocities
@jit(nopython=True)
def numba_random_seed(seed):
    """Same as np.random.seed but for numba. Function adapated from velocyto.


    Parameters
    ----------
        seed: `int`
            Random seed value

    Returns
    -------
        Nothing but set the random seed in numba.

    """
    np.random.seed(seed)


@jit(nopython=True)
def permute_rows_nsign(A):
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently. Function adapted
    from velocyto

    Parameters
    ----------
        A: `np.array`
            A numpy array that will be permuted.

    Returns
    -------
        Nothing but permute entries and signs for each row of the input matrix in place.
    """

    plmi = np.array([+1, -1])
    for i in range(A.shape[1]):
        np.random.shuffle(A[:, i])
        A[:, i] = A[:, i] * np.random.choice(plmi, size=A.shape[0])


def embed_velocity(adata, x_basis, v_basis='velocity', emb_basis='X', velocity_gene_tag='velocity_genes',
                   num_pca=100, n_recurse_neighbors=2, M_diff=0.25, adaptive_local_kernel=True, normalize_velocity=True,
                   return_kmc=False, **kmc_kwargs):
    if velocity_gene_tag is not None:
        X = adata.layers[x_basis][:, adata.var[velocity_gene_tag]]
        V = adata.layers[v_basis][:, adata.var[velocity_gene_tag]]
    else:
        X = adata.layers[x_basis]
        V = adata.layers[v_basis]

    X = log1p_(adata, X)

    X_emb = adata.obsm[emb_basis]
    Idx = adata.uns['neighbors']['indices']

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
    kmc.fit(X_pca[:, :num_pca], V_pca[:, :num_pca], neighbor_idx=Idx,
            n_recurse_neighbors=n_recurse_neighbors, M_diff=M_diff, adaptive_local_kernel=adaptive_local_kernel,
            **kmc_kwargs)

    Uc = kmc.compute_density_corrected_drift(X_emb, normalize_vector=normalize_velocity)
    if return_kmc:
        return Uc, kmc
    else:
        return Uc
