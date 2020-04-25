import numpy as np
from .utils import vector_field_function, integrate_vf_ivp
from .utils import fetch_states


def fate(
    adata,
    init_cells,
    init_states=None,
    basis=None,
    layer="X",
    genes=None,
    t_end=None,
    direction="both",
    average="origin",
    VecFld_true=None,
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
            Cell name or indices of the initial cell states for the historical or future cell state prediction with numerical integration.
            If the names in init_cells are not find in the adata.obs_name, it will be treated as cell indices and must be integers.
        init_states: `numpy.ndarray` or None (default: None)
            Initial cell states for the historical or future cell state prediction with numerical integration.
        basis: `str` or None (default: `None`)
            The embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the reconstructed trajectory
            will be projected back to high dimensional space via the `inverse_transform` function.
        layer: `str` or None (default: 'X')
            Which layer of the data will be used for predicting cell fate with the reconstructed vector field function.
            The layer once provided, will override the `basis` argument and then predicting cell fate in high dimensional space.
        genes: `list` or None (default: None)
            The gene names whose gene expression will be used for predicting cell fate. By default (when genes is set to None),
            the genes used for velocity embedding (var.use_for_velocity) will be used for vector field reconstruction. Note that
            the genes to be used need to have velocity calculated and corresponds to those used in the `dyn.tl.VectorField` function.
        t_end: `float` (default None)
            The length of the time period from which to predict cell state forward or backward over time. This is used
            by the odeint function.
        direction: `string` (default: both)
            The direction to predict the cell fate. One of the `forward`, `backward` or `both` string.
        average: `str` (default: `origin`) {'origin', 'trajectory'}
            The method to calculate the average cell state at each time step, can be one of `origin` or `trajectory`. If `origin` used,
            the average expression state from the init_cells will be calculated and the fate prediction is based on this state. If `trajectory`
            used, the average expression states of all cells predicted from the vector field function at each time point will be used.
        VecFld_true: `function`
            The true ODE function, useful when the data is generated through simulation. Replace VecFld arugment when this has been set.
        kwargs:
            Additional parameters that will be passed into the fate function.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the dictionary Fate (includes `t` and `prediction` keys) in uns attribute.
    """

    if basis is not None:
        fate_key = "fate_" + basis
    else:
        fate_key = "fate" if layer == "X" else "fate_" + layer

    init_states, VecFld, t_end, valid_genes = fetch_states(
        adata, init_states, init_cells, basis, layer, average, t_end
    )

    t, prediction = _fate(
        VecFld,
        init_states,
        VecFld_true=VecFld_true,
        direction=direction,
        t_end=t_end,
        average=True,
        **kwargs
    )

    high_prediction = None
    if basis == "pca":
        high_prediction = adata.uns["pca_fit"].inverse_transform(prediction)
        if adata.var.use_for_dynamo.sum() == high_prediction.shape[1]:
            valid_genes = adata.var_names[adata.var.use_for_dynamo]
        else:
            valid_genes = adata.var_names[adata.var.use_for_velocity]

    elif basis == "umap":
        # this requires umap 0.4
        high_prediction = adata.uns["umap_fit"].inverse_transform(prediction)
        ndim = adata.uns["umap_fit"]._raw_data.shape[1]

        if "X" in adata.obsm_keys():
            if ndim == adata.obsm["X"].shape[1]:  # lift the dimension up again
                high_prediction = adata.uns["pca_fit"].inverse_transform(prediction)

        if adata.var.use_for_dynamo.sum() == high_prediction.shape[1]:
            valid_genes = adata.var_names[adata.var.use_for_dynamo]
        elif adata.var.use_for_velocity.sum() == high_prediction.shape[1]:
            valid_genes = adata.var_names[adata.var.use_for_velocity]
        else:
            raise Exception(
                "looks like a customized set of genes is used for pca analysis of the adata. "
                "Try rerunning pca analysis with default settings for this function to work."
            )

    if VecFld_true is None:
        adata.uns[fate_key] = (
            {
                "init_states": init_states,
                "average": average,
                "t": t,
                "prediction": prediction,
                "VecFld_true": VecFld_true,
                "genes": valid_genes,
            }
            if high_prediction is None
            else {
                "init_states": init_states,
                "average": average,
                "t": t,
                "prediction": prediction,
                "high_prediction": high_prediction,
                "VecFld_true": VecFld_true,
                "genes": valid_genes,
            }
        )
    else:
        adata.uns[fate_key] = (
            {
                "init_states": init_states,
                "average": average,
                "t": t,
                "prediction": prediction,
                "genes": valid_genes,
            }
            if high_prediction is None
            else {
                "init_states": init_states,
                "average": average,
                "t": t,
                "prediction": prediction,
                "high_prediction": high_prediction,
                "genes": valid_genes,
            }
        )

    return adata


def _fate(
    VecFld,
    init_states,
    VecFld_true=None,
    t_end=None,
    step_size=None,
    direction="both",
    interpolation_num=250,
    average=True,
):
    """Predict the historical and future cell transcriptomic states over arbitrary time scales by integrating vector field
    functions from one or a set of initial cell state(s).

    Arguments
    ---------
        VecFld: `function`
            Functional form of the vector field reconstructed from sparse single cell samples. It is applicable to the entire
            transcriptomic space.
        init_states: `numpy.ndarray`
            Initial cell states for the historical or future cell state prediction with numerical integration.
        VecFld_true: `function` or `None`
            The true ODE function, useful when the data is generated through simulation. Replace VecFld arugment when this has been set.
        t_end: `float` (default None)
            The length of the time period from which to predict cell state forward or backward over time. This is used
            by the odeint function.
        step_size: `float` or None (default None)
            Step size for integrating the future or history cell state, used by the odeint function. By default it is None,
            and the step_size will be automatically calculated to ensure 250 total integration time-steps will be used.
        direction: `string` (default: both)
            The direction to predict the cell fate. One of the `forward`, `backward`or `both` string.
        interpolation_num: `int` (default: 250)
            The number of uniformly interpolated time points.
        average: `bool` (default: True)
            X boolean flag to determine whether to smooth the trajectory by calculating the average cell state at each time
            step.

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

    V_func = (
        lambda x: vector_field_function(x=x, t=None, VecFld=VecFld)
        if VecFld_true is None
        else VecFld_true
    )

    if step_size is None:
        max_steps = (
            int(max(7 / (init_states.shape[1] / 300), 4))
            if init_states.shape[1] > 300
            else 7
        )
        t_linspace = np.linspace(
            0, t_end, 10 ** (np.min([int(np.log10(t_end)), max_steps]))
        )
    else:
        t_linspace = np.arange(0, t_end + step_size, step_size)

    t, prediction = integrate_vf_ivp(
        init_states,
        t_linspace,
        (),
        direction,
        V_func,
        interpolation_num=interpolation_num,
        average=average,
    )
    return t, prediction


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
