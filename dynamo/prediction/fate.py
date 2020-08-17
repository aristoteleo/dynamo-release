from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from multiprocessing.dummy import Pool as ThreadPool
import itertools
import warnings
from .utils import (
    integrate_vf_ivp,
    fetch_states,
    getTseq,
)
from ..vectorfield import vector_field_function


def fate(
    adata,
    init_cells,
    init_states=None,
    basis=None,
    layer="X",
    dims=None,
    genes=None,
    t_end=None,
    direction="both",
    interpolation_num=250,
    average=False,
    sampling='arc_length',
    VecFld_true=None,
    inverse_transform=False,
    scale=1,
    cores=1,
    **kwargs
):
    """Predict the historical and future cell transcriptomic states over arbitrary time scales.

     This is achieved by integrating the reconstructed vector field function from one or a set of initial cell state(s).
     Note that this function is designed so that there is only one trajectory (based on averaged cell states if multiple
     initial states are provided) will be returned. `dyn.tl._fate` can be used to calculate multiple cell states.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        init_cells: `list` (default: None)
            Cell name or indices of the initial cell states for the historical or future cell state prediction with
            numerical integration. If the names in init_cells are not find in the adata.obs_name, it will be treated as
            cell indices and must be integers.
        init_states: `numpy.ndarray` or None (default: None)
            Initial cell states for the historical or future cell state prediction with numerical integration.
        basis: `str` or None (default: `None`)
            The embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the reconstructed
            trajectory will be projected back to high dimensional space via the `inverse_transform` function.
        layer: `str` or None (default: 'X')
            Which layer of the data will be used for predicting cell fate with the reconstructed vector field function.
            The layer once provided, will override the `basis` argument and then predicting cell fate in high dimensional
            space.
        genes: `list` or None (default: None)
            The gene names whose gene expression will be used for predicting cell fate. By default (when genes is set to
            None), the genes used for velocity embedding (var.use_for_velocity) will be used for vector field
            reconstruction. Note that the genes to be used need to have velocity calculated and corresponds to those used
            in the `dyn.tl.VectorField` function.
        t_end: `float` (default None)
            The length of the time period from which to predict cell state forward or backward over time. This is used
            by the odeint function.
        direction: `string` (default: both)
            The direction to predict the cell fate. One of the `forward`, `backward` or `both` string.
        interpolation_num: `int` (default: 100)
            The number of uniformly interpolated time points.
        average: `str` or `bool` (default: `False`) {'origin', 'trajectory'}
            The method to calculate the average cell state at each time step, can be one of `origin` or `trajectory`. If
            `origin` used, the average expression state from the init_cells will be calculated and the fate prediction is
            based on this state. If `trajectory` used, the average expression states of all cells predicted from the
            vector field function at each time point will be used. If `average` is `False`, no averaging will be applied.
        sampling: `str` (default: `arc_length`)
            Methods to sample points along the integration path, one of `{'arc_length', 'logspace', 'uniform_indices'}`.
            If `logspace`, we will sample time points linearly on log space. If `uniform_indices`, the sorted unique set
            of all time points from all cell states' fate prediction will be used and then evenly sampled up to
            `interpolation_num` time points. If `arc_length`, we will sample the integration path with uniform arc length.
        VecFld_true: `function`
            The true ODE function, useful when the data is generated through simulation. Replace VecFld arugment when
            this has been set.
        inverse_transform: `bool` (default: `False`)
            Whether to inverse transform the low dimensional vector field prediction back to high dimensional space.
        scale: `float` (default: `1`)
            The value that will be used to scale the predicted velocity value from the reconstructed vector field function.
        cores: `int` (default: 1):
            Number of cores to calculate path integral for predicting cell fate. If cores is set to be > 1,
            multiprocessing will be used to parallel the fate prediction.
        kwargs:
            Additional parameters that will be passed into the fate function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the dictionary Fate (includes `t` and `prediction` keys) in uns attribute.
    """

    if sampling in ['arc_length', 'logspace', 'uniform_indices']:
        if average in ['origin', 'trajectory', True]:
            warnings.warn(
               f"using {sampling} to sample data points along an integral path at different integration "
                "time points. Average trajectory won't be calculated")

        average = False

    if basis is not None:
        fate_key = "fate_" + basis
        #vf_key = "VecFld_" + basis
    else:
        fate_key = "fate" if layer == "X" else "fate_" + layer
        #vf_key = "VecFld"

    #VecFld = adata.uns[vf_key]["VecFld"]
    #X = VecFld["X"]
    #xmin, xmax = X.min(0), X.max(0)
    #t_end = np.max(xmax - xmin) / np.min(np.abs(VecFld["V"]))
    #valid_genes = None

    init_states, VecFld, t_end, valid_genes = fetch_states(
        adata, init_states, init_cells, basis, layer, True if average in ['origin', 'trajectory', True] else False, t_end
    )

    if np.isscalar(dims):
        init_states = init_states[:, :dims]
    elif dims is not None:
        init_states = init_states[:, dims]

    vf = lambda x: scale*vector_field_function(x=x, vf_dict=VecFld, dim=dims) if VecFld_true is None else VecFld_true
    t, prediction = _fate(
        vf,
        init_states,
        t_end=t_end,
        direction=direction,
        interpolation_num=interpolation_num,
        average=True if average in ['origin', 'trajectory', True] else False,
        sampling=sampling,
        cores=cores,
        **kwargs
    )

    high_prediction = None
    if basis == "pca" and inverse_transform:
        high_prediction = adata.uns["pca_fit"].inverse_transform(prediction)
        if adata.var.use_for_dynamics.sum() == high_prediction.shape[1]:
            valid_genes = adata.var_names[adata.var.use_for_dynamics]
        else:
            valid_genes = adata.var_names[adata.var.use_for_velocity]

    elif basis == "umap" and inverse_transform:
        # this requires umap 0.4; reverse project to PCA space.
        if prediction.ndim == 1: prediction = prediction[None, :]
        high_prediction = adata.uns["umap_fit"]['fit'].inverse_transform(prediction)

        # further reverse project back to raw expression space
        PCs = adata.uns['PCs'].T
        if PCs.shape[0] == high_prediction.shape[1]:
            high_prediction = high_prediction @ PCs

        ndim = adata.uns["umap_fit"]['fit']._raw_data.shape[1]

        if "X" in adata.obsm_keys():
            if ndim == adata.obsm["X"].shape[1]:  # lift the dimension up again
                high_prediction = adata.uns["pca_fit"].inverse_transform(prediction)

        if adata.var.use_for_dynamics.sum() == high_prediction.shape[1]:
            valid_genes = adata.var_names[adata.var.use_for_dynamics]
        elif adata.var.use_for_velocity.sum() == high_prediction.shape[1]:
            valid_genes = adata.var_names[adata.var.use_for_velocity]
        else:
            raise Exception(
                "looks like a customized set of genes is used for pca analysis of the adata. "
                "Try rerunning pca analysis with default settings for this function to work."
            )

    adata.uns[fate_key] = {
            "init_states": init_states,
            "init_cells": init_cells,
            "average": average,
            "t": t,
            "prediction": prediction,
            # "VecFld": VecFld,
            "VecFld_true": VecFld_true,
            "genes": valid_genes,
        }
    if high_prediction is not None:
        adata.uns[fate_key]["inverse_transform"] = high_prediction

    return adata


def _fate(
    VecFld,
    init_states,
    t_end=None,
    step_size=None,
    direction="both",
    interpolation_num=250,
    average=True,
    sampling='arc_length',
    cores=1,
):
    """Predict the historical and future cell transcriptomic states over arbitrary time scales by integrating vector field
    functions from one or a set of initial cell state(s).

    Arguments
    ---------
        VecFld: `function`
            Functional form of the vector field reconstructed from sparse single cell samples. It is applicable to the
            entire transcriptomic space.
        init_states: `numpy.ndarray`
            Initial cell states for the historical or future cell state prediction with numerical integration.
        t_end: `float` (default None)
            The length of the time period from which to predict cell state forward or backward over time. This is used
            by the odeint function.
        step_size: `float` or None (default None)
            Step size for integrating the future or history cell state, used by the odeint function. By default it is None,
            and the step_size will be automatically calculated to ensure 250 total integration time-steps will be used.
        direction: `string` (default: both)
            The direction to predict the cell fate. One of the `forward`, `backward`or `both` string.
        interpolation_num: `int` (default: 100)
            The number of uniformly interpolated time points.
        average: `bool` (default: True)
            A boolean flag to determine whether to smooth the trajectory by calculating the average cell state at each
            time step.
        sampling: `str` (default: `logspace`)
            Methods to sample points along the integration path, one of `{'arc_length', 'logspace', 'uniform_indices'}`.
            If `logspace`, we will sample time points linearly on log space. If `uniform_indices`, the sorted unique set
            of all time points from all cell states' fate prediction will be used and then evenly sampled up to
            `interpolation_num` time points. If `arc_length`, we will sample the integration path with uniform arc length.
        cores: `int` (default: 1):
            Number of cores to calculate path integral for predicting cell fate. If cores is set to be > 1,
            multiprocessing will be used to parallel the fate prediction.

    Returns
    -------
    t: `numpy.ndarray`
        The time at which the cell state are predicted.
    prediction: `numpy.ndarray`
        Predicted cells states at different time points. Row order corresponds to the element order in t. If init_states
        corresponds to multiple cells, the expression dynamics over time for each cell is concatenated by rows. That is,
        the final dimension of prediction is (len(t) * n_cells, n_features). n_cells: number of cells; n_features: number
        of genes or number of low dimensional embeddings. Of note, if the average is set to be True, the average cell state
        at each time point is calculated for all cells.
    """

    t_linspace = getTseq(init_states, t_end, step_size)

    if cores == 1:
        t, prediction = integrate_vf_ivp(
            init_states,
            t_linspace,
            (),
            direction,
            VecFld,
            interpolation_num=interpolation_num,
            average=average,
            sampling=sampling,
        )
    else:
        pool = ThreadPool(cores)
        res = pool.starmap(integrate_vf_ivp, zip(init_states, itertools.repeat(t_linspace), itertools.repeat(()),
                                      itertools.repeat(direction), itertools.repeat(VecFld),
                                      itertools.repeat(interpolation_num), itertools.repeat(False),
                                      itertools.repeat(True))) # disable tqdm when using multiple cores.
        pool.close()
        pool.join()
        t_, prediction_ = zip(*res)
        t, prediction = [i[0] for i in t_], [i[0] for i in prediction_]
        t, prediction = np.hstack(t), np.hstack(prediction)
        n_cell, n_feature = init_states.shape
        if init_states.shape[0] > 1 and average:
            t_len = int(len(t) / n_cell)
            avg = np.zeros((n_feature, t_len))

            for i in range(t_len):
                avg[:, i] = np.mean(prediction[:, np.arange(n_cell) * t_len + i], 1)

            prediction = avg
            t = np.sort(np.unique(t))

    return t, prediction


def fate_bias(adata,
              group,
              basis='umap',
              inds=None,
              speed_percentile=5,
              dist_threshold=None,
              source_groups=None,
              metric="euclidean",
              metric_kwds=None,
              cores=1,
              seed=19491001,
              **kwargs,
            ):
    """Calculate the lineage (fate) bias of states whose trajectory are predicted.

    Fate bias is currently calculated as the percentage of points along the predicted cell fate trajectory whose distance
    to their 0-th nearest neighbors on the data are close enough (determined by median 1-st nearest neighbors of all
    observed cells and the dist_threshold) to any cell from each group specified by `group` key. The details is described
    as following:

    Cell fate predicted by our vector field method sometimes end up in regions that are not sampled with cells. We thus
    developed a heuristic method to iteratively walk backward the integration path to assign cell fate. We first identify
    the regions with small velocity in the tail of the integration path (determined by `speed_percentile`), then we check
    whether the distance of 0-th nearest points on the observed data to all those points are far away from the observed
    data (determined by `dist_threshold`). If they are not all close to data, we then walk backwards along the trajectory
    by one time step until the distance of any currently visited integration path’s data points’ 0-th nearest points to
    the observed cells is close enough. In order to calculate the cell fate probability, we diffuse one step further of
    the identified nearest neighbors from the integration to identify more nearest observed cells, especially those from
    terminal cell types in case nearby cells first identified are all close to some random progenitor cells. Then we use
    group information of those observed cells to define the fate probability.

    `fate_bias` calculate a confidence score for the calculated fate probability with a simple metric, defined as
        :math:`1 - (sum(distances > dist_threshold * median_dist) + walk_back_steps) / (len(indices) + walk_back_steps)`

    The `distance` is currently visited integration path’s data points’ 0-th nearest points to the observed cells.
    `median_dist` is median distance of their 1-st nearest cell distance of all observed cells. `walk_back_steps` is the
    steps walked backward along the integration path until all currently visited integration points's 0-th nearest points
    to the observed cells satisfy the distance threshold. `indices` are the time indices of integration points that is
    regarded as the regions with `small velocity` (note when walking backward, those corresponding points are not
    necessarily have small velocity anymore).

    Arguments
    ---------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the predicted fate trajectories in the `uns` attribute.
        group: `str`
            The column key that corresponds to the cell type or other group information for quantifying the bias of cell
            state.
        basis: `str` or None (default: `None`)
            The embedding data space where cell fates were predicted and cell fates bias will be quantified.
        inds `list` or `float` or None (default: `None`):
            The indices of the time steps that will be used for calculating fate bias. If inds is None, the last a few
            steps of the fate prediction based on the `sink_speed_percentile` will be use. If inds is the float (between
            0 and 1), it will be regarded as a percentage, and the last percentage of steps will be used for fate bias
            calculation. Otherwise inds need to be a list of integers of the time steps.
        speed_percentile: `float` (default: `5`)
            The percentile of speed that will be used to determine the terminal cells (or sink region on the prediction
            path where speed is smaller than this speed percentile).
        dist_threshold: `float` or `None` (default: `None`)
            A multiplier of the median nearest cell distance on the embedding to determine cells that are outside the
            sampled domain of cells. If the mean distance of identified "terminal cells" is above this number, we will
            look backward along the trajectory (by minimize all indices by 1) until it finds cells satisfy this threshold.
            By default it is set to be 1 to ensure only considering points that are very close to observed data points.
        source_groups: `list` or `None` (default: `None`)
            The groups that corresponds to progenitor groups. They has to have at least one intersection with the groups
            from the `group` column. If group is not `None`, any identified "source_groups" cells that happen to be in
            those groups will be ignored and the probability of cell fate of those cells will be reassigned to the group
            that has the highest fate probability among other non source_groups group cells.
        metric: `str` or callable, default='euclidean'
            The distance metric to use for the tree.  The default metric is , and with p=2 is equivalent to the standard
            Euclidean metric. See the documentation of :class:`DistanceMetric` for a list of available metrics. If metric
            is "precomputed", X is assumed to be a distance matrix and must be square during fit. X may be a
            :term:`sparse graph`, in which case only "nonzero" elements may be considered neighbors.
        metric_kwds : dict, default=None
            Additional keyword arguments for the metric function.
        cores: `int` (default: 1)
            The number of parallel jobs to run for neighbors search. ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.
        seed: `int` (default `19491001`)
            Random seed to ensure the reproducibility of each run.
        kwargs:
            Additional arguments that will be passed to each nearest neighbor search algorithm.

    Returns
    -------
        fate_bias: `pandas.DataFrame`
            A DataFrame that stores the fate bias for each cell state (row) to each cell group (column).
    """

    if dist_threshold is None:
        dist_threshold = 1

    if group not in adata.obs.keys():
        raise ValueError(f'The group {group} you provided is not a key of .obs attribute.')
    else:
        clusters = adata.obs[group]

    basis_key = 'X_' + basis if basis is not None else 'X'
    fate_key = 'fate_' + basis if basis is not None else 'fate'

    if basis_key not in adata.obsm.keys():
        raise ValueError(f'The basis {basis_key} you provided is not a key of .obsm attribute.')
    if fate_key not in adata.uns.keys():
        raise ValueError(f"The {fate_key} key is not existed in the .uns attribute of the adata object. You need to run"
                         f"dyn.pd.fate(adata, basis='{basis}') before calculate fate bias.")

    if source_groups is not None:
        if type(source_groups) is str: source_groups = [source_groups]
        source_groups = list(set(source_groups).intersection(clusters))
        if len(source_groups) == 0:
            raise ValueError(f"the {source_groups} you provided doesn't intersect with any groups in the {group} column.")

    X = adata.obsm[basis_key] if basis_key != 'X' else adata.X

    if X.shape[0] > 5000 and X.shape[1] > 2:
        from pynndescent import NNDescent

        nbrs = NNDescent(X, metric=metric, metric_kwds=metric_kwds, n_neighbors=30, n_jobs=cores,
                              random_state=seed, **kwargs)
        knn, distances = nbrs.query(X, k=30)
    else:
        alg = 'ball_tree' if X.shape[1] > 10 else 'kd_tree'
        nbrs = NearestNeighbors(n_neighbors=30, algorithm=alg, n_jobs=cores).fit(X)
        distances, knn = nbrs.kneighbors(X)

    median_dist = np.median(distances[:, 1])

    pred_dict = {}
    cell_predictions, cell_indx = adata.uns[fate_key]['prediction'], adata.uns[fate_key]['init_cells']
    t = adata.uns[fate_key]['t']
    confidence = np.zeros(len(t))

    for i, prediction in tqdm(enumerate(cell_predictions), desc='calculating fate distributions'):
        cur_t, n_steps = t[i], len(t[i])

        # ensure to identify sink where the speed is very slow if inds is not provided.
        # if inds is the percentage, use the last percentage of steps to check for cell fate bias.
        # otherwise inds need to be a list.
        if inds is None:
            avg_speed = np.array([np.linalg.norm(i) for i in np.diff(prediction, 1).T]) / np.diff(cur_t)
            sink_checker = np.where(avg_speed[::-1] > np.percentile(avg_speed, speed_percentile))[0]
            indices = np.arange(n_steps - max(min(sink_checker), 10), n_steps)
        elif inds is float:
            indices = np.arange(int(n_steps - inds * n_steps), n_steps)
        else:
            indices = inds

        if hasattr(nbrs, 'query'):
            knn, distances = nbrs.query(prediction[:, indices].T, k=30) 
        else:
            distances, knn = nbrs.kneighbors(prediction[:, indices].T) 

        # if final steps too far away from observed cells, ignore them
        walk_back_steps = 0
        while True:
            is_dist_larger_than_threshold = distances.flatten() < dist_threshold * median_dist
            if any(is_dist_larger_than_threshold):

                # let us diffuse one step further to identify cells from terminal cell types in case
                # cells with indices are all close to some random progenitor cells.
                if hasattr(nbrs, 'query'):
                    knn, _ = nbrs.query(X[knn.flatten(), :], k=30)
                else:
                    _, knn = knn, nbrs.kneighbors(X[knn.flatten(), :])

                fate_prob = clusters[knn.flatten()].value_counts() / len(knn.flatten())
                if source_groups is not None:
                    source_p = fate_prob[source_groups].sum()
                    if 1 > source_p > 0:
                        fate_prob[source_groups] = 0
                        fate_prob[fate_prob.idxmax()] += source_p

                pred_dict[i] = fate_prob

                confidence[i] = 1 - (sum(~ is_dist_larger_than_threshold) + walk_back_steps) / (
                        len(indices) + walk_back_steps)

                break
            else:
                walk_back_steps += 1

                if any(indices - 1 < 0):
                    pred_dict[i] = clusters[knn.flatten()].value_counts() * np.nan
                    break

                if hasattr(nbrs, 'query'):
                    knn, distances = nbrs.query(prediction[:, indices - 1].T, k=30)
                else:
                    distances, knn = nbrs.kneighbors(prediction[:, indices - 1].T) 

                knn, distances = knn[:, 0], distances[:, 0]
                indices = indices - 1

    bias = pd.DataFrame(pred_dict).T
    conf = pd.DataFrame({"confidence": confidence}, index=bias.index)
    bias = pd.merge(conf, bias, left_index=True, right_index=True)

    if cell_indx is not None: bias.index = cell_indx

    return bias

# def fate_(adata, time, direction = 'forward'):
#     from .moments import *
#     gene_exprs = adata.X
#     cell_num, gene_num = gene_exprs.shape
#
#
#     for i in range(gene_num):
#         params = {'a': adata.uns['dynamo'][i, "a"], \
#                   'b': adata.uns['dynamo'][i, "b"], \
#                   'la': adata.uns['dynamo'][i, "la"], \
#                   'alpha_a': adata.uns['dynamo'][i, "alpha_a"], \
#                   'alpha_i': adata.uns['dynamo'][i, "alpha_i"], \
#                   'sigma': adata.uns['dynamo'][i, "sigma"], \
#                   'beta': adata.uns['dynamo'][i, "beta"], \
#                   'gamma': adata.uns['dynamo'][i, "gamma"]}
#         mom = moments_simple(**params)
#         for j in range(cell_num):
#             x0 = gene_exprs[i, j]
#             mom.set_initial_condition(*x0)
#             if direction == "forward":
#                 gene_exprs[i, j] = mom.solve([0, time])
#             elif direction == "backward":
#                 gene_exprs[i, j] = mom.solve([0, - time])
#
#     adata.uns['prediction'] = gene_exprs
#     return adata
