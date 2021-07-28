from typing import Union
from hdbscan import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
from anndata import AnnData
import pandas as pd
from anndata import AnnData
from ..dynamo_logger import main_info
from ..utils import LoggerManager, copy_adata
from ..tools.clustering import (
    hdbscan,
    leiden,
    louvain,
    infomap,
)
from ..tools.Markov import (
    velocity_on_grid,
    grid_velocity_filter,
    prepare_velocity_grid_data,
)
from .utils import vecfld_from_adata


def cluster_field(
    adata,
    basis="pca",
    features=["speed", "potential", "divergence", "acceleration", "curvature", "curl"],
    add_embedding_basis=True,
    embedding_basis=None,
    normalize=True,
    method="leiden",
    cores=1,
    copy=False,
    **kwargs,
):
    """Cluster cells based on vector field features.

    We would like to see whether the vector field can be used to better define cell state/types. This can be accessed
    via characterizing critical points (attractor/saddle/repressor, etc.) and characteristic curves (nullcline,
    separatrix). However, the calculation of those is not easy, for example, a strict definition of an attractor is
    states where velocity is 0 and the eigenvalue of the jacobian matrix at that point is all negative. Under this
    strict definition, we may sometimes find the attractors are very far away from our sampled cell states which makes
    them less meaningful although this can be largely avoided when we decide to remove the density correction during the
    velocity projection. This is not unexpected as the vector field we learned is defined via a set of basis functions
    based on gaussian kernels and thus it is hard to satisfy that strict definition.

    Fortunately, we can handle this better with the help of a different set of ideas. Instead of using critical points
    by the classical dynamic system methods, we can use some machine learning approaches that are based on extracting
    geometric features of streamline to "cluster vector field space" for define cell states/type. This requires
    calculating, potential (ordered pseudotime), speed, curliness, divergence, acceleration, curvature, etc. Thanks to
    the fact that we can analytically calculate Jacobian matrix matrix, those quantities of the vector field function
    can be conveniently and efficiently calculated.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`.
        adata object that includes both newly synthesized and total gene expression of cells. Alternatively,
        the object should include both unspliced and spliced gene expression of cells.
    basis: `str` or None (default: `None`)
        The space that will be used for calculating vector field features. Valid names includes, for example, `pca`,
        `umap`, etc.
    embedding_basis: `str` or None (default: `None`)
        The embedding basis that will be combined with the vector field feature space for clustering.
    normalize: `bool` (default: `True`)
        Whether to mean center and scale the feature across all cells so that the mean
    method: `str` (default: `leiden`)
        The method that will be used for clustering, one of `{'kmeans'', 'hdbscan', 'louvain', 'leiden'}`. If `louvain`
        or `leiden` used, you need to have `cdlib` installed.
    cores: `int` (default: 1)
        The number of parallel jobs to run for neighbors search. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    copy:
        Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments.
    kwargs:
        Any additional arguments that will be passed to either kmeans, hdbscan, louvain or leiden clustering algorithms.

    Returns
    -------

    """

    logger = LoggerManager.gen_logger("dynamo-cluster_field")
    logger.log_time()
    adata = copy_adata(adata) if copy else adata

    if method in ["louvain", "leiden"]:
        try:
            from cdlib import algorithms

            "leiden" in dir(algorithms)

        except ImportError:
            raise ImportError(
                "You need to install the excellent package `cdlib` if you want to use louvain or leiden "
                "for clustering."
            )

    feature_key = [
        "speed" + basis,
        basis + "_ddhodge_potential",
        "divergence_" + basis,
        "acceleration_" + basis,
        "curvature_" + basis,
        "curl_" + basis,
    ]
    feature_list = [i + "_" + basis if i != "potential" else basis + "_" + i for i in features]

    if feature_key[0] not in adata.obs.keys() and feature_key[0] in feature_list:
        from ..vectorfield import speed

        speed(adata, basis=basis)
    if feature_key[1] not in adata.obs.keys() and feature_key[1] in feature_list:
        from ..ext import ddhodge

        ddhodge(adata, basis=basis)
    if feature_key[2] not in adata.obs.keys() and feature_key[2] in feature_list:
        from ..vectorfield import divergence

        divergence(adata, basis=basis)
    if feature_key[3] not in adata.obs.keys() and feature_key[3] in feature_list:
        from ..vectorfield import acceleration

        acceleration(adata, basis=basis)
    if feature_key[4] not in adata.obs.keys() and feature_key[4] in feature_list:
        from ..vectorfield import curvature

        curvature(adata, basis=basis)

    if feature_key[5] not in adata.obs.keys() and feature_key[5] in feature_list:
        from ..vectorfield import curl

        curl(adata, basis=basis)

    feature_data = adata.obs.loc[:, feature_key].values
    if embedding_basis is None:
        embedding_basis = basis
    if add_embedding_basis:
        X = np.hstack((feature_data, adata.obsm["X_" + embedding_basis]))
    else:
        X = feature_data

    if normalize:
        # X = (X - X.min(0)) / X.ptp(0)
        X = (X - X.mean(0)) / X.std(0)

    if method in ["hdbscan", "kmeans"]:
        if method == "hdbscan":
            key = "field_hdbscan"
            hdbscan(adata, X_data=X, result_key=key, **kwargs)
        elif method == "kmeans":
            from sklearn.cluster import KMeans

            key = "field_kmeans"

            kmeans = KMeans(random_state=0, **kwargs).fit(X)
            adata.obs[key] = kmeans.labels_.astype("str")

        # clusters need to be categorical variables
        adata.obs.obs[key] = adata.obs.obs[key].astype("category")

    elif method in ["louvain", "leiden"]:
        if X.shape[0] > 200000 and X.shape[1] > 2:
            from pynndescent import NNDescent

            nbrs = NNDescent(
                X,
                metric="euclidean",
                n_neighbors=31,
                n_jobs=cores,
                random_state=19491001,
            )
            nbrs_idx, dist = nbrs.query(X, k=31)
        else:
            nbrs = NearestNeighbors(n_neighbors=31, n_jobs=cores).fit(X)
            dist, nbrs_idx = nbrs.kneighbors(X)

        row = np.repeat(nbrs_idx[:, 0], 30)
        col = nbrs_idx[:, 1:].flatten()
        graph = csr_matrix(
            (np.repeat(1, len(col)), (row, col)),
            shape=(adata.n_obs, adata.n_obs),
        )
        adata.obsp["vf_feature_knn"] = graph

        if method == "leiden":
            leiden(
                adata,
                adj_matrix_key="vf_feature_knn",
                result_key="field_leiden",
            )
        elif method == "louvain":
            louvain(
                adata,
                adj_matrix_key="vf_feature_knn",
                result_key="field_louvain",
            )
        elif method == "infomap":
            infomap(
                adata,
                adj_matrix_key="vf_feature_knn",
                result_key="field_infomap",
            )

    logger.finish_progress(progress_name="clustering_field")

    if copy:
        return adata
    return None


def streamline_clusters(
    adata: AnnData,
    basis: str = "umap",
    method: str = "gaussian",
    xy_grid_nums: list = [50, 50],
    density: float = 5,
    clustering_method: str = "leiden",
):
    import matplotlib.pyplot as plt

    if method in ["louvain", "leiden"]:
        try:
            from cdlib import algorithms

            "leiden" in dir(algorithms)

        except ImportError:
            raise ImportError(
                "You need to install the excellent package `cdlib` if you want to use louvain or leiden "
                "for clustering."
            )

    vf_dict = adata.uns["VecFld_" + basis]

    X_grid, V_grid = (
        vf_dict["grid"],
        vf_dict["grid_V"],
    )
    N = int(np.sqrt(V_grid.shape[0]))

    grid_kwargs_dict = {
        "density": None,
        "smooth": None,
        "n_neighbors": None,
        "min_mass": None,
        "autoscale": False,
        "adjust_for_stream": True,
        "V_threshold": None,
    }

    if method.lower() == "sparsevfc":
        X, V = adata.obsm["X_" + basis], adata.obsm["velocity_" + basis]
        X_grid, p_mass, neighs, weight = prepare_velocity_grid_data(
            X,
            xy_grid_nums,
            density=grid_kwargs_dict["density"],
            smooth=grid_kwargs_dict["smooth"],
            n_neighbors=grid_kwargs_dict["n_neighbors"],
        )
        for i in ["density", "smooth", "n_neighbors"]:
            grid_kwargs_dict.pop(i)

        VecFld, func = vecfld_from_adata(adata, basis)

        V_emb = func(X)
        V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1) / np.maximum(1, p_mass)[:, None]
        X_grid, V_grid = grid_velocity_filter(
            V_emb=V,
            neighs=neighs,
            p_mass=p_mass,
            X_grid=X_grid,
            V_grid=V_grid,
            **grid_kwargs_dict,
        )
    elif method.lower() == "gaussian":
        X_grid, V_grid, D = velocity_on_grid(
            vf_dict["X"],
            vf_dict["Y"],
            xy_grid_nums,
            cut_off_velocity=True,
            **grid_kwargs_dict,
        )
    else:
        raise ValueError(f"only `sparsevfc` and `gaussian` method supported")

    strm = plt.streamplot(
        X_grid[0],
        X_grid[1],
        V_grid[0],
        V_grid[1],
        density=density,
    )
    strm_res = strm.lines.get_segments()  # np.array(strm.lines.get_segments()).reshape((-1, 2))

    line_list_ori = {}
    line_ind = 0
    for i, seg in enumerate(strm_res):
        if i == 0:
            line_list_ori[0] = [seg]
        else:
            if all(strm_res[i - 1][1] == seg[0]):
                line_list_ori[line_ind].append(seg)
            else:
                line_ind += 1
                line_list_ori[line_ind] = [seg]

    line_list = line_list_ori.copy()
    for key, values in line_list_ori.items():
        line_list_ori[key] = np.array(values).reshape((-1, 2))

    for key, values in line_list.items():
        line_list[key] = np.unique(np.array(values).reshape((-1, 2)), axis=0)

    from dynamo.vectorfield.scVectorField import SvcVectorfield

    vector_field_class = SvcVectorfield()
    vector_field_class.from_adata(adata, basis=basis)

    acc_dict = {}
    cur_1_dict = {}
    cur_2_dict = {}
    div_dict = {}
    speed_dict = {}
    curl_dict = {}

    for key, values in line_list.items():
        acceleration_val, acceleration_vec = vector_field_class.compute_acceleration(values)
        curvature_val_1 = vector_field_class.compute_curvature(values, formula=1)
        curvature_val_2, curvature_vec = vector_field_class.compute_curvature(values)
        divergence_val = vector_field_class.compute_divergence(values)
        speed_vec = vector_field_class.func(values)
        speed_val = np.linalg.norm(speed_vec)
        curl_val = vector_field_class.compute_curl(values)

        acc_dict[key] = acceleration_val
        cur_1_dict[key] = curvature_val_1
        cur_2_dict[key] = curvature_val_2
        div_dict[key] = divergence_val
        speed_dict[key] = speed_val
        curl_dict[key] = curl_val

    # create histogram
    bins = 10  # 10 bins
    line_len = []
    feature_df = np.zeros((len(line_list), 6 * bins))
    for key, values in line_list.items():
        line_len.append(values.shape[0])
        _, acc_hist = np.histogram(acc_dict[key], bins=(bins - 1), density=True)
        _, cur_1_hist = np.histogram(cur_1_dict[key][0], bins=(bins - 1), density=True)
        _, cur_2_hist = np.histogram(cur_2_dict[key], bins=(bins - 1), density=True)
        _, div_hist = np.histogram(div_dict[key], bins=(bins - 1), density=True)
        _, speed_hist = np.histogram(speed_dict[key], bins=(bins - 1), density=True)
        _, curl_hist = np.histogram(curl_dict[key], bins=(bins - 1), density=True)

        feature_df[key, :] = np.hstack((acc_hist, cur_1_hist, cur_2_hist, div_hist, speed_hist, curl_hist))

    from ..preprocessing.utils import pca

    feature_adata = AnnData(feature_df)
    pca(feature_adata, X_data=feature_df, pca_key="X_pca")
    if clustering_method == "louvain":
        louvain(feature_adata, obsm_key="X_pca")
    elif clustering_method == "leiden":
        leiden(feature_adata, obsm_key="X_pca")
    elif clustering_method == "infomap":
        infomap(feature_adata, obsm_key="X_pca")
    elif clustering_method == "leiden":
        leiden(feature_adata, obsm_key="X_pca")

    adata.uns["streamline_clusters_" + basis] = {
        "feature_df": feature_df,
        "segments": line_list_ori,
        "clustering_method": clustering_method,
        "clusters": adata.obs[clustering_method],
    }
