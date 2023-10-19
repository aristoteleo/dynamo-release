from typing import List, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors

from ..dynamo_logger import main_info
from ..preprocessing.pca import pca
from ..tools.clustering import hdbscan, leiden, louvain
from ..tools.connectivity import k_nearest_neighbors
from ..tools.Markov import (
    grid_velocity_filter,
    prepare_velocity_grid_data,
    velocity_on_grid,
)
from ..utils import LoggerManager, copy_adata
from .scVectorField import SvcVectorField
from .utils import vecfld_from_adata


def cluster_field(
    adata: AnnData,
    basis: str = "pca",
    features: List = ["speed", "potential", "divergence", "acceleration", "curvature", "curl"],
    add_embedding_basis: bool = True,
    embedding_basis: Optional[str] = None,
    normalize: bool = False,
    method: str = "leiden",
    cores: int = 1,
    copy: bool = False,
    resolution: float = 1.0,
    **kwargs,
) -> Optional[AnnData]:
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
    the fact that we can analytically calculate the Jacobian matrix, those quantities of the vector field function
    can be conveniently and efficiently calculated.

    Args:
        adata: adata object that includes both newly synthesized and total gene expression of cells. Alternatively,
            the object should include both unspliced and spliced gene expression of cells.
        basis: The space that will be used for calculating vector field features. Valid names includes, for example, `pca`,
            `umap`, etc.
        features: features have to be selected from ['speed', 'potential', 'divergence', 'acceleration', 'curvature', 'curl']
        add_embedding_basis: Whether to add the embedding basis to the feature space for clustering.
        embedding_basis: The embedding basis that will be combined with the vector field feature space for clustering.
        normalize: Whether to mean center and scale the feature across all cells.
        method: The method that will be used for clustering, one of `{'kmeans'', 'hdbscan', 'louvain', 'leiden'}`. If `louvain`
            or `leiden` used, you need to have `cdlib` installed.
        cores: The number of parallel jobs to run for neighbors search. ``None`` means 1 unless in a
            :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.
        copy: Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments.
        resolution: Clustering resolution, higher values yield more fine-grained clusters.
        kwargs: Any additional arguments that will be passed to either kmeans, hdbscan, louvain or leiden clustering algorithms.

    Returns:
        Either updates `adata` or directly returns a new `adata` object if `copy` is `True`.

    """

    logger = LoggerManager.gen_logger("dynamo-cluster_field")
    logger.log_time()
    adata = copy_adata(adata) if copy else adata

    features = list(
        set(features).intersection(["speed", "potential", "divergence", "acceleration", "curvature", "curl"])
    )
    if len(features) < 1:
        raise ValueError(
            "features have to be selected from ['speed', 'potential', 'divergence', 'acceleration', "
            f"'curvature', 'curl']. your feature is {features}"
        )

    feature_key = [
        "speed_" + basis,
        basis + "_ddhodge_potential",
        "divergence_" + basis,
        "acceleration_" + basis,
        "curvature_" + basis,
        "curl_" + basis,
    ]
    feature_list = [i + "_" + basis if i != "potential" else basis + "_ddhodge_" + i for i in features]

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

    feature_data = adata.obs.loc[:, feature_list].values
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
        adata.obs[key] = adata.obs.obs[key].astype("category")

    elif method in ["louvain", "leiden"]:
        nbrs_idx, dist = k_nearest_neighbors(
            X,
            k=30,
            exclude_self=False,
            pynn_rand_state=19491001,
            n_jobs=cores,
            logger=logger,
        )

        row = np.repeat(nbrs_idx[:, 0], 30)
        col = nbrs_idx[:, 1:].flatten()
        graph = csr_matrix(
            (np.repeat(1, len(col)), (row, col)),
            shape=(adata.n_obs, adata.n_obs),
        )
        adata.obsp["vf_feature_knn"] = graph

        if method == "leiden":
            leiden(adata, resolution=resolution, adj_matrix_key="vf_feature_knn", result_key="field_leiden", **kwargs)
        elif method == "louvain":
            louvain(adata, resolution=resolution, adj_matrix_key="vf_feature_knn", result_key="field_louvain", **kwargs)

    logger.finish_progress(progress_name="clustering_field")

    if copy:
        return adata
    return None


def streamline_clusters(
    adata: AnnData,
    basis: str = "umap",
    features: list = ["speed", "divergence", "acceleration", "curvature", "curl"],
    method: str = "sparsevfc",
    xy_grid_nums: list = [50, 50],
    density: float = 5,
    curvature_method: int = 1,
    feature_bins: int = 10,
    clustering_method: str = "leiden",
    assign_fixedpoints: bool = False,
    reversed_fixedpoints: bool = False,
    **kwargs,
) -> None:
    """Cluster 2D streamlines based on vector field features. Initialize a grid over the state space and compute the
    flow of data through the grid using plt.streamplot with a given density. For each point individual streamline,
    computes the vector field 'features' of interest and stores the data via histograms. Add fixed points and
    "reversed fixed points" (sources of the streamlines) to the feature data dataframe based on the
    'assigned_fixedpoints' and 'reversed_fixedpoints' args. Finally, then cluster the streamlines based on these
    features using the given 'clustering_method'.

    Args:
        adata: An AnnData object representing the network to be analyzed.
        basis: The basis to use for creating the vector field, either "umap" or "tsne". Defaults to "umap".
        features: A list of features to calculate for each point in the vector field. Defaults to ["speed", "divergence", "acceleration", "curvature", "curl"].
        method: The method to use for calculating the flow of data through the grid, either "sparsevfc" or "gaussian". Defaults to "sparsevfc".
        xy_grid_nums: The number of points to use in the x and y dimensions of the grid. Defaults to [50, 50].
        density: The density of the grid. Defaults to 5.
        curvature_method: The method to use for calculating curvature. Defaults to 1.
        feature_bins: The number of bins to use for discretizing the data. Defaults to 10.
        clustering_method: The method to use for clustering the data into modules, either "louvain" or "leiden". Defaults to "leiden".
        assign_fixedpoints: A boolean indicating whether to assign fixed points to the data. Defaults to False.
        reversed_fixedpoints: A boolean indicating whether to reverse the fixed points assignment. Defaults to False.

    Raises:
        ImportError: If the "cdlib" package is not installed and the "louvain" or "leiden" clustering method is specified.
        ValueError: If an invalid method is specified for calculating the flow of data through the grid.
        ValueError: If an invalid method is specified for clustering the data into modules.

    Returns:
        None, but updates the `adata` object with the following fields of the `adata.uns["streamline_clusters_" + basis]`
            -  "feature_df"
            - "segments"
            - "X_pca"
            - "clustering_method"
            - "distances"
            - "connectivities"
            - "clusters"
            - "fixed_point"
            - "rev_fixed_point"
    """

    import matplotlib.pyplot as plt

    vf_dict, func = vecfld_from_adata(adata, basis=basis)
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
    strm_res = strm.lines.get_segments()  # get streamline segements

    # split segments into different streamlines
    line_list_ori = {}
    line_ind = 0
    for i, seg in enumerate(strm_res):
        if i == 0:
            line_list_ori[0] = [seg]
        else:
            # the second point from the previous segment should be the same from the first point in the current segment
            if all(strm_res[i - 1][1] == seg[0]):
                line_list_ori[line_ind].append(seg)
            else:
                line_ind += 1
                line_list_ori[line_ind] = [seg]

    line_list = line_list_ori.copy()

    # convert to list of numpy arrays.
    for key, values in line_list_ori.items():
        line_list_ori[key] = np.array(values).reshape((-1, 2))

    # remove duplicated rows from the numpy arrays.
    for key, values in line_list.items():
        line_list[key] = np.unique(np.array(values).reshape((-1, 2)), axis=0)

    vector_field_class = SvcVectorField()
    vector_field_class.from_adata(adata, basis=basis)

    has_acc = True if "acceleration" in features else False
    has_curv = True if "curvature" in features else False
    has_div = True if "divergence" in features else False
    has_speed = True if "speed" in features else False
    has_curl = True if "curl" in features else False

    if has_acc:
        acc_dict = {}
    if has_curv:
        cur_1_dict = {}
        cur_2_dict = {}
    if has_div:
        div_dict = {}
    if has_speed:
        speed_dict = {}
    if has_curl:
        curl_dict = {}

    # save features along the streameline and create histogram for each feature
    bins = feature_bins  # number of feature bins
    line_len = []
    feature_df = np.zeros((len(line_list), len(features) * bins))

    for key, values in line_list.items():
        line_len.append(values.shape[0])
        tmp = None
        if has_acc:
            acceleration_val, _ = vector_field_class.compute_acceleration(values)
            acc_dict[key] = acceleration_val

            _, acc_hist = np.histogram(acceleration_val, bins=(bins - 1), density=True)
            if tmp is None:
                tmp = acc_hist
        if has_curv:
            curvature_val_1 = vector_field_class.compute_curvature(values, formula=1)[0]
            cur_1_dict[key] = curvature_val_1

            curvature_val_2, curvature_vec = vector_field_class.compute_curvature(values)
            cur_2_dict[key] = curvature_val_2

            _, cur_1_hist = np.histogram(curvature_val_1, bins=(bins - 1), density=True)
            _, cur_2_hist = np.histogram(curvature_val_2, bins=(bins - 1), density=True)
            if tmp is None:
                tmp = cur_1_hist if curvature_method == 1 else cur_2_hist
            else:
                tmp = np.hstack((tmp, cur_1_hist if curvature_method == 1 else cur_2_hist))
        if has_div:
            divergence_val = vector_field_class.compute_divergence(values)
            div_dict[key] = divergence_val

            _, div_hist = np.histogram(divergence_val, bins=(bins - 1), density=True)
            if tmp is None:
                tmp = div_hist
            else:
                tmp = np.hstack((tmp, div_hist))
        if has_speed:
            speed_vec = vector_field_class.func(values)
            speed_val = np.linalg.norm(speed_vec)
            speed_dict[key] = speed_val

            _, speed_hist = np.histogram(speed_val, bins=(bins - 1), density=True)
            if tmp is None:
                tmp = speed_hist
            else:
                tmp = np.hstack((tmp, speed_hist))
        if has_curl:
            curl_val = vector_field_class.compute_curl(values)
            curl_dict[key] = curl_val

            _, curl_hist = np.histogram(curl_val, bins=(bins - 1), density=True)
            if tmp is None:
                tmp = curl_hist
            else:
                tmp = np.hstack((tmp, curl_hist))

        feature_df[key, :] = tmp

    # clustering
    feature_adata = AnnData(feature_df)
    pca(feature_adata, X_data=feature_df, pca_key="X_pca")
    if clustering_method == "louvain":
        louvain(feature_adata, obsm_key="X_pca")
    elif clustering_method == "leiden":
        leiden(feature_adata, obsm_key="X_pca")
    elif method in ["hdbscan", "kmeans"]:
        key = "field_hdbscan"
        hdbscan(feature_adata, X_data=feature_df, result_key=key, **kwargs)
    elif method == "kmeans":
        from sklearn.cluster import KMeans

        key = "field_kmeans"
        kmeans = KMeans(random_state=0, **kwargs).fit(X)
        feature_adata.obs[key] = kmeans.labels_.astype("str")

        # clusters need to be categorical variables
        feature_adata.obs[key] = adata.obs.obs[key].astype("category")
    else:
        raise ValueError(
            "only louvain, leiden, hdbscan and kmeans clustering supported but your requested "
            f"method is {method}"
        )

    if assign_fixedpoints or reversed_fixedpoints:
        tmp = np.array(strm.lines.get_segments()).reshape((-1, 2))
        vector_field_class.data["X"] = np.unique(tmp, axis=0)

        if assign_fixedpoints:
            (
                X,
                valid_fps_type_assignment,
                assignment_id,
            ) = vector_field_class.assign_fixed_points(cores=1)

            feature_adata.obs["fixed_point"] = -1

        if reversed_fixedpoints:
            # reverse vector field to identify source:
            vector_field_class.func = lambda x: -vector_field_class.func(x)
            (
                X_rev,
                valid_fps_type_assignment_rev,
                assignment_id_rev,
            ) = vector_field_class.assign_fixed_points(cores=1)

            feature_adata.obs["rev_fixed_point"] = -1

        data_X = vector_field_class.data["X"]
        for key, values in line_list.items():
            indices = [np.where(np.logical_and(data_X[:, 0] == val[0], data_X[:, 1] == val[1]))[0][0] for val in values]

            # assign fixed point to the most frequent point
            if assign_fixedpoints:
                mode_val = mode(assignment_id[indices])[0][0]
                if not np.isnan(mode_val):
                    feature_adata.obs.loc[str(key), "fixed_point"] = mode_val
            if reversed_fixedpoints:
                mode_val = mode(assignment_id_rev[indices])[0][0]
                if not np.isnan(mode_val):
                    feature_adata.obs.loc[str(key), "rev_fixed_point"] = mode_val

    adata.uns["streamline_clusters_" + basis] = {
        "feature_df": feature_df,
        "segments": line_list_ori,
        "X_pca": feature_adata.obsm["X_pca"],
        "clustering_method": clustering_method,
        "distances": feature_adata.obsp["X_pca_distances"],
        "connectivities": feature_adata.obsp["X_pca_connectivities"],
        "clusters": feature_adata.obs[clustering_method].values,
    }

    if assign_fixedpoints:
        adata.uns["streamline_clusters_" + basis]["fixed_point"] = feature_adata.obs["fixed_point"]
    if reversed_fixedpoints:
        adata.uns["streamline_clusters_" + basis]["rev_fixed_point"] = feature_adata.obs["rev_fixed_point"]
