# create by Yan Zhang, minor adjusted by Xiaojie Qiu
import datetime
import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import scipy.sparse as sp

from ..dynamo_logger import LoggerManager, main_info, main_warning
from ..tools.utils import inverse_norm, update_dict
from ..utils import copy_adata
from .scVectorField import BaseVectorField, SvcVectorField
from .topography import topography
from .utils import angle


def VectorField(
    adata: anndata.AnnData,
    basis: Optional[str] = None,
    layer: Optional[str] = None,
    dims: Optional[Union[int, list]] = None,
    genes: Optional[list] = None,
    normalize: Optional[bool] = False,
    grid_velocity: bool = False,
    grid_num: int = 50,
    velocity_key: str = "velocity_S",
    method: str = "SparseVFC",
    min_vel_corr: float = 0.6,
    restart_num: int = 5,
    restart_seed: Optional[list] = [0, 100, 200, 300, 400],
    model_buffer_path: Optional[str] = None,
    return_vf_object: bool = False,
    map_topography: bool = False,
    pot_curl_div: bool = False,
    cores: int = 1,
    result_key: Optional[str] = None,
    copy: bool = False,
    n: int = 25,
    **kwargs,
) -> Union[anndata.AnnData, BaseVectorField]:
    """Learn a function of high dimensional vector field from sparse single cell samples in the entire space robustly.

    Args:
        adata: AnnData object that contains embedding and velocity data
        basis: The embedding data to use. The vector field function will be learned on the low  dimensional embedding
            and can be then projected back to the high dimensional space.
        layer: Which layer of the data will be used for vector field function reconstruction. The layer once provided,
            will override the `basis` argument and then learn the vector field function in high dimensional space.
        dims: The dimensions that will be used for reconstructing vector field functions. If it is an `int` all
            dimension from the first dimension to `dims` will be used; if it is a list, the dimensions in the list will
            be used.
        genes: The gene names whose gene expression will be used for vector field reconstruction. By default, (when
            genes is set to None), the genes used for velocity embedding (var.use_for_transition) will be used for
            vector field reconstruction. Note that the genes to be used need to have velocity calculated.
        normalize: Logic flag to determine whether to normalize the data to have zero means and unit covariance. This is
            often required for raw dataset (for example, raw UMI counts and RNA velocity values in high dimension). But
            it is normally not required for low dimensional embeddings by PCA or other non-linear dimension reduction
            methods.
        grid_velocity: Whether to generate grid velocity. Note that by default it is set to be False, but for datasets
            with embedding dimension less than 4, the grid velocity will still be generated. Please note that number of
            total grids in the space increases exponentially as the number of dimensions increases. So it may quickly
            lead to lack of memory, for example, it cannot allocate the array with grid_num set to be 50 and dimension
            is 6 (50^6 total grids) on 32 G memory computer. Although grid velocity may not be generated, the vector
            field function can still be learned for thousands of dimensions and we can still predict the transcriptomic
            cell states over long time period.
        grid_num: The number of grids in each dimension for generating the grid velocity.
        velocity_key: The key from the adata layer that corresponds to the velocity matrix.
        method: Method that is used to reconstruct the vector field functionally. Currently only SparseVFC supported but
            other improved approaches are under development.
        min_vel_corr: The minimal threshold for the cosine correlation between input velocities and learned velocities
            to consider as a successful vector field reconstruction procedure. If the cosine correlation is less than
            this threshold and restart_num > 1, `restart_num` trials will be attempted with different seeds to
            reconstruct the vector field function. This can avoid some reconstructions to be trapped in some local
            optimal.
        restart_num: The number of retrials for vector field reconstructions.
        restart_seed: A list of seeds for each retrial. Must be the same length as `restart_num` or None.
        model_buffer_path: The directory address keeping all the saved/to-be-saved torch variables and NN
            modules. When `method` is set to be `dynode`, buffer_path will be constructed with working directory,
            `basis` and datetime.
        return_vf_object: Whether to include an instance of a vectorfield class in the `VecFld` dictionary in the
            `uns`attribute.
        map_topography: Whether to quantify the topography of vector field. Note that for higher than 2D vector field,
            we can only identify fixed points as high-dimensional nullcline and separatrices are mathematically
            difficult to be identified. Nullcline and separatrices will also be a surface or manifold in
            high-dimensional vector field.
        pot_curl_div: Whether to calculate potential, curl or divergence for each cell. Potential can be calculated for
            any basis while curl and divergence is by default only applied to 2D basis. However, divergence is
            applicable for any dimension while curl is generally only defined for 2/3 D systems.
        cores: Number of cores to run the ddhodge function. If cores is set to be > 1, multiprocessing will be used to
            parallel the ddhodge calculation.
        result_key:
            The key that will be used as prefix for the vector field key in .uns
        copy: Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments and
            returning `None`.
        n: Number of samples for calculating the fixed points.
        kwargs: Other additional parameters passed to the vectorfield class.

    Returns:
        If `copy` and `return_vf_object` arguments are set to False, `annData` object is updated with the
            `VecFld`dictionary in the `uns` attribute.
        If `return_vf_object` is set to True, then a vector field class object is returned.
        If `copy` is set to True, a deep copy of the original `adata` object is returned.
    """
    logger = LoggerManager.gen_logger("dynamo-topography")
    logger.info("VectorField reconstruction begins...", indent_level=1)
    logger.log_time()
    adata = copy_adata(adata) if copy else adata

    X, V, valid_genes = _get_X_V_for_VectorField(
        adata,
        basis=basis,
        layer=layer,
        dims=dims,
        genes=genes,
        velocity_key=velocity_key,
        logger=logger,
    )

    Grid = None
    if X.shape[1] < 4 or grid_velocity:
        Grid = _generate_grid(X, grid_num=grid_num, logger=logger)

    if X is None:
        raise Exception(f"X is None. Make sure you passed the correct X or {basis} dimension reduction method.")
    elif V is None:
        raise Exception("V is None. Make sure you passed the correct V.")

    logger.info("Learning vector field with method: %s." % (method.lower()))

    Dynode_obj = None
    if method.lower() == "sparsevfc":
        vf_kwargs = _get_svc_default_arguments(**kwargs)
        VecFld = SvcVectorField(X, V, Grid, normalize=normalize, **vf_kwargs)
        train_kwargs = kwargs
    elif method.lower() == "dynode":
        try:
            from dynode.vectorfield import Dynode  # networkModels,

            from .scVectorField import dynode_vectorfield
        except ImportError:
            raise ImportError("You need to install the package `dynode`." "install dynode via `pip install dynode`")

        if not ("Dynode" in kwargs and type(kwargs["Dynode"]) == Dynode):
            vf_kwargs, train_kwargs = _get_dynode_default_arguments(
                X,
                V,
                basis=basis,
                normalize=normalize,
                model_buffer_path=model_buffer_path,
                **kwargs,
            )
            VecFld = dynode_vectorfield(X, V, Grid, **vf_kwargs)
        else:
            Dynode_obj = kwargs["Dynode"]
            vf_kwargs, train_kwargs = {}, {}
            VecFld = dynode_vectorfield.fromDynode(Dynode_obj)
    else:
        raise ValueError("current only support two methods, SparseVFC and dynode")

    if restart_num > 0:
        vf_dict = _resume_training(
            VecFld=VecFld,
            train_kwargs=train_kwargs,
            method=method,
            min_vel_corr=min_vel_corr,
            restart_num=restart_num,
            restart_seed=restart_seed,
            Dynode_obj=Dynode_obj,
        )
    else:
        vf_dict = VecFld.train(**train_kwargs)

    if result_key is None:
        vf_key = "VecFld" if basis is None else "VecFld_" + basis
    else:
        vf_key = result_key if basis is None else result_key + "_" + basis

    vf_dict["method"] = method
    if basis is not None:
        key = "velocity_" + basis + "_" + method
        X_copy_key = "X_" + basis + "_" + method

        logger.info_insert_adata(key, adata_attr="obsm")
        logger.info_insert_adata(X_copy_key, adata_attr="obsm")
        adata.obsm[key] = vf_dict["V"]
        adata.obsm[X_copy_key] = vf_dict["X"]

        vf_dict["dims"] = dims

        logger.info_insert_adata(vf_key, adata_attr="uns")
        adata.uns[vf_key] = vf_dict
    else:
        key = velocity_key + "_" + method

        logger.info_insert_adata(key, adata_attr="layers")
        adata.layers[key] = sp.csr_matrix((adata.shape))
        adata.layers[key][:, [adata.var_names.get_loc(i) for i in valid_genes]] = vf_dict["V"]

        vf_dict["layer"] = layer
        vf_dict["genes"] = genes
        vf_dict["velocity_key"] = velocity_key

        logger.info_insert_adata(vf_key, adata_attr="uns")
        adata.uns[vf_key] = vf_dict

    if map_topography:
        tp_kwargs = {"n": n}
        tp_kwargs = update_dict(tp_kwargs, kwargs)

        logger.info("Mapping topography...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            adata = topography(
                adata,
                basis=basis,
                X=X,
                layer=layer,
                dims=None,
                VecFld=vf_dict,
                **tp_kwargs,
            )
    if pot_curl_div:
        from .vector_calculus import curl, divergence

        logger.info(f"Running ddhodge to estimate vector field based pseudotime in {basis} basis...")
        from ..external.hodge import ddhodge

        ddhodge(adata, basis=basis, cores=cores)
        if X.shape[1] == 2:
            logger.info("Computing curl...")
            curl(adata, basis=basis)

        logger.info("Computing divergence...")
        divergence(adata, basis=basis)

    control_point, inlier_prob, valid_ids = (
        "control_point_" + basis if basis is not None else "control_point",
        "inlier_prob_" + basis if basis is not None else "inlier_prob",
        vf_dict["valid_ind"],
    )
    if method.lower() == "sparsevfc":
        logger.info_insert_adata(control_point, adata_attr="obs")
        logger.info_insert_adata(inlier_prob, adata_attr="obs")

        adata.obs[control_point], adata.obs[inlier_prob] = False, np.nan
        adata.obs.loc[adata.obs_names[vf_dict["ctrl_idx"]], control_point] = True
        adata.obs.loc[adata.obs_names[valid_ids], inlier_prob] = vf_dict["P"].flatten()

    # angles between observed velocity and that predicted by vector field across cells:
    cell_angles = np.zeros(adata.n_obs, dtype=float)
    for i, u, v in zip(valid_ids, V[valid_ids], vf_dict["V"]):
        # fix the u, v norm == 0 in angle function
        cell_angles[i] = angle(u.astype("float64"), v.astype("float64"))

    if basis is not None:
        temp_key = "obs_vf_angle_" + basis

        logger.info_insert_adata(temp_key, adata_attr="obs")
        adata.obs[temp_key] = cell_angles
    else:
        temp_key = "obs_vf_angle"
        logger.info_insert_adata(temp_key, adata_attr="obs")
        adata.obs[temp_key] = cell_angles

    logger.finish_progress("VectorField")
    if return_vf_object:
        return VecFld
    elif copy:
        return adata
    return None


def _get_svc_default_arguments(**kwargs) -> Dict:
    """Get default arguments for vector field learning with SparseVFC method."""
    vf_kwargs = {
        "M": None,
        "a": 5,
        "beta": None,
        "ecr": 1e-5,
        "gamma": 0.9,
        "lambda_": 3,
        "minP": 1e-5,
        "MaxIter": 30,
        "theta": 0.75,
        "div_cur_free_kernels": False,
        "velocity_based_sampling": True,
        "sigma": 0.8,
        "eta": 0.5,
        "seed": 0,
    }
    vf_kwargs = update_dict(vf_kwargs, kwargs)

    return vf_kwargs


def _get_dynode_default_arguments(
    X: np.ndarray,
    V: np.ndarray,
    basis: Optional[str] = None,
    normalize: Optional[bool] = False,
    model_buffer_path: Optional[str] = None,
    **kwargs,
) -> Tuple[Dict, Dict]:
    """Get default arguments for vector field learning with dynode method."""
    try:
        from dynode.vectorfield import networkModels
        from dynode.vectorfield.losses_weighted import MSE
        from dynode.vectorfield.samplers import VelocityDataSampler
    except ImportError:
        raise ImportError("You need to install the package `dynode`." "install dynode via `pip install dynode`")

    velocity_data_sampler = VelocityDataSampler(adata={"X": X, "V": V}, normalize_velocity=normalize)
    max_iter = 2 * 100000 * np.log(X.shape[0]) / (250 + np.log(X.shape[0]))

    cwd, cwt = os.getcwd(), datetime.datetime.now()

    if model_buffer_path is None:
        model_buffer_path = cwd + "/" + basis + "_" + str(cwt.year) + "_" + str(cwt.month) + "_" + str(cwt.day)
        main_warning("the buffer path saving the dynode model is in %s" % (model_buffer_path))

    vf_kwargs = {
        "model": networkModels,
        "sirens": False,
        "enforce_positivity": False,
        "velocity_data_sampler": velocity_data_sampler,
        "time_course_data_sampler": None,
        "network_dim": X.shape[1],
        "velocity_loss_function": MSE(),  # CosineDistance(), # #MSE(), MAD()
        # BinomialChannel(p=0.1, alpha=1)
        "time_course_loss_function": None,
        "velocity_x_initialize": X,
        "time_course_x0_initialize": None,
        "smoothing_factor": None,
        "stability_factor": None,
        "load_model_from_buffer": False,
        "buffer_path": model_buffer_path,
        "hidden_features": 256,
        "hidden_layers": 3,
        "first_omega_0": 30.0,
        "hidden_omega_0": 30.0,
    }
    train_kwargs = {
        "max_iter": int(max_iter),
        "velocity_batch_size": 50,
        "time_course_batch_size": 100,
        "autoencoder_batch_size": 50,
        "velocity_lr": 1e-4,
        "velocity_x_lr": 0,
        "time_course_lr": 1e-4,
        "time_course_x0_lr": 1e4,
        "autoencoder_lr": 1e-4,
        "velocity_sample_fraction": 1,
        "time_course_sample_fraction": 1,
        "iter_per_sample_update": None,
    }

    vf_kwargs = update_dict(vf_kwargs, kwargs)
    train_kwargs = update_dict(train_kwargs, kwargs)

    return vf_kwargs, train_kwargs


def _get_X_V_for_VectorField(
    adata: anndata.AnnData,
    basis: Optional[str] = None,
    layer: Optional[str] = None,
    dims: Optional[Union[int, list]] = None,
    genes: Optional[list] = None,
    velocity_key: str = "velocity_S",
    logger: Optional[LoggerManager] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Get X and V for vector field reconstruction."""
    if basis is not None:
        logger.info(
            "Retrieve X and V based on basis: %s. \n "
            "       Vector field will be learned in the %s space." % (basis.upper(), basis.upper())
        )
        X = adata.obsm["X_" + basis].copy()
        V = adata.obsm["velocity_" + basis].copy()

        if np.isscalar(dims):
            X, V = X[:, :dims], V[:, :dims]
        elif type(dims) is list:
            X, V = X[:, dims], V[:, dims]

        valid_genes = adata.var.index

    else:
        logger.info(
            "Retrieve X and V based on `genes`, layer: %s. \n "
            "       Vector field will be learned in the gene expression space." % layer
        )
        valid_genes = (
            list(set(genes).intersection(adata.var.index))
            if genes is not None
            else adata.var_names[adata.var.use_for_transition]
        )
        if layer == "X":
            X = adata[:, valid_genes].X.copy()
            X = np.expm1(X)
        else:
            X = inverse_norm(adata, adata.layers[layer])

        V = adata[:, valid_genes].layers[velocity_key].copy()

        if sp.issparse(X):
            X, V = X.A, V.A

        # keep only genes with finite velocity and expression values, useful when learning vector field in the original
        # gene expression space.
        finite_genes = np.logical_and(np.isfinite(X).all(axis=0), np.isfinite(V).all(axis=0))
        X, V = X[:, finite_genes], V[:, finite_genes]
        valid_genes = np.array(valid_genes)[np.where(finite_genes)[0]].tolist()
        if sum(finite_genes) < len(finite_genes):
            logger.warning(
                f"There are {(len(finite_genes) - sum(finite_genes))} genes with infinite expression or velocity "
                f"values. These genes will be excluded from vector field reconstruction. Please make sure the genes you "
                f"selected has no non-infinite values"
            )

    return X, V, valid_genes


def _generate_grid(
    X: np.ndarray,
    grid_num: int = 50,
    logger: Optional[LoggerManager] = None,
) -> np.ndarray:
    """Generate high dimensional grids and convert into a row matrix for vector field reconstruction."""
    logger.info("Generating high dimensional grids and convert into a row matrix.")

    min_vec, max_vec = (
        X.min(0),
        X.max(0),
    )
    min_vec = min_vec - 0.01 * np.abs(max_vec - min_vec)
    max_vec = max_vec + 0.01 * np.abs(max_vec - min_vec)

    Grid_list = np.meshgrid(*[np.linspace(i, j, grid_num) for i, j in zip(min_vec, max_vec)])
    Grid = np.array([i.flatten() for i in Grid_list]).T

    return Grid


def _resume_training(
    VecFld: BaseVectorField,
    train_kwargs: Dict,
    method: str,
    min_vel_corr: float,
    restart_num: int,
    restart_seed: Optional[list] = [0, 100, 200, 300, 400],
    Dynode_obj: Optional[BaseVectorField] = None,
) -> Dict:
    """Resume vector field reconstruction from given restart_num and restart_seed."""
    if len(restart_seed) != restart_num:
        main_warning(
            f"the length of {restart_seed} is different from {restart_num}, " f"using `np.range(restart_num) * 100"
        )
        restart_seed = np.arange(restart_num) * 100
    restart_counter, cur_vf_list, res_list = 0, [], []
    while True:
        if Dynode_obj is None:
            if method.lower() == "sparsevfc":
                train_kwargs.update({"seed": restart_seed[restart_counter]})
            cur_vf_dict = VecFld.train(**train_kwargs)
        else:
            X, Y = Dynode_obj.Velocity["sampler"].X_raw, Dynode_obj.Velocity["sampler"].V_raw
            cur_vf_dict = {
                "X": X,
                "Y": Y,
                "V": Dynode_obj.predict_velocity(Dynode_obj.Velocity["sampler"].X_raw),
                "grid_V": Dynode_obj.predict_velocity(Dynode_obj.Velocity["sampler"].Grid),
                "valid_ind": (
                    Dynode_obj.Velocity["sampler"].valid_ind
                    if hasattr(Dynode_obj.Velocity["sampler"], "valid_ind")
                    else np.arange(X.shape[0])
                ),
                "parameters": Dynode_obj.Velocity,
                "dynode_object": VecFld,
            }

        # consider refactor with .simulation.evaluation.py
        reference, prediction = (
            cur_vf_dict["Y"][cur_vf_dict["valid_ind"]],
            cur_vf_dict["V"][cur_vf_dict["valid_ind"]],
        )
        true_normalized = reference / (np.linalg.norm(reference, axis=1).reshape(-1, 1) + 1e-20)
        predict_normalized = prediction / (np.linalg.norm(prediction, axis=1).reshape(-1, 1) + 1e-20)
        res = np.mean(true_normalized * predict_normalized) * prediction.shape[1]

        cur_vf_list += [cur_vf_dict]
        res_list += [res]
        if res < min_vel_corr:
            restart_counter += 1
            main_info(
                f"current cosine correlation between input velocities and learned velocities is less than "
                f"{min_vel_corr}. Make a {restart_counter}-th vector field reconstruction trial.",
                indent_level=2,
            )
        else:
            vf_dict = cur_vf_dict
            break

        if restart_counter > restart_num - 1:
            main_warning(
                f"Cosine correlation between input velocities and learned velocities is less than"
                f" {min_vel_corr} after {restart_num} trials of vector field reconstruction."
            )
            vf_dict = cur_vf_list[np.argmax(np.array(res_list))]

            break

    return vf_dict
