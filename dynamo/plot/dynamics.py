import numpy as np
import pandas as pd
import sys
import warnings
from scipy.sparse import issparse
from .utils import despline, _matplotlib_points, _datashade_points, _select_font_color
from .utils import quiver_autoscaler, default_quiver_args
from .utils import save
from .scatters import scatters
from ..tools.velocity import sol_u, sol_s, solve_first_order_deg
from ..tools.utils_moments import moments
from ..tools.utils import get_mapper, one_shot_k
from ..tools.utils import update_dict
from ..configuration import _themes, set_figure_params


def phase_portraits(
    adata,
    genes,
    x=0,
    y=1,
    pointsize=None,
    vkey="S",
    ekey="X",
    basis="umap",
    color=None,
    figsize=(7, 5),
    ncols=None,
    legend="upper left",
    background=None,
    show_quiver=False,
    quiver_size=None,
    quiver_length=None,
    q_kwargs_dict={},
    save_show_or_return='show',
    save_kwargs={},
    **kwargs,
):
    """Draw the phase portrait, velocity, expression values on the low dimensional embedding.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        genes: `list`
            A list of gene names that are going to be visualized.
        x: `int` (default: `0`)
                The column index of the low dimensional embedding for the x-axis
        y: `int` (default: `1`)
                The column index of the low dimensional embedding for the y-axis
        pointsize: `None` or `float` (default: None)
                The scale of the point size. Actual point cell size is calculated as `500.0 / np.sqrt(adata.shape[0]) * pointsize`
        vkey: `string` (default: velocity)
            Which velocity key used for visualizing the magnitude of velocity. Can be either velocity in the layers slot or the
            keys in the obsm slot.
        ekey: `str`
            The layer of data to represent the gene expression level.
        basis: `string` (default: umap)
            Which low dimensional embedding will be used to visualize the cell.
        color: `string` (default: None)
            Which group will be used to color cells, only used for the phase portrait because the other two plots are colored
            by the velocity magnitude or the gene expression value, respectively.
        figsize: `None` or `[float, float]` (default: None)
                The width and height of a figure.
        ncols: `None` or `int` (default: None)
        ncol: `None` or `int` (default: None)
                Number of columns in each facet grid.
        legend: `str` (default: `on data`)
                Where to put the legend.  Legend is drawn by seaborn with “brief” mode, numeric hue and size variables will be
                represented with a sample of evenly spaced values. By default legend is drawn on top of cells.
        show_quiver: `bool` (default: False)
            Whether to show the quiver plot. If velocity for x component (corresponds to either spliced, total RNA, protein, etc)
             or y component (corresponds to either unspliced, new RNA, protein, etc) are both calculated, quiver represents
             velocity for both components otherwise the uncalculated component (usually y component) will be set to be 0.
        quiver_size: `float` or None (default: None)
            The size of quiver. If None, we will use set quiver_size to be 1. Note that quiver quiver_size is used to calculate
            the head_width (10 x quiver_size), head_length (12 x quiver_size) and headaxislength (8 x quiver_size) of the quiver.
            This is done via the `default_quiver_args` function which also calculate the scale of the quiver (1 / quiver_length).
        quiver_length: `float` or None (default: None)
            The length of quiver. The quiver length which will be used to calculate scale of quiver. Note that befoe applying
            `default_quiver_args` velocity values are first rescaled via the quiver_autoscaler function. Scale of quiver indicates
            the nuumber of data units per arrow length unit, e.g., m/s per plot width; a smaller scale parameter makes the arrow longer.
        q_kwargs_dict: `dict` (default: {})
            The dictionary of the quiver arguments. The default setting of quiver argument is identical to that used in the
            cell_wise_velocity and grid_velocity.
        save_show_or_return: {'show', 'save', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save function. By default it is an empty dictionary and the save function
            will use the {"path": None, "prefix": 'phase_portraits', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.
        **kwargs:
            Additional parameters that will be passed to plt.scatter function

    Returns
    -------
        A matplotlib plot that shows 1) the phase portrait of each category used in velocity embedding, cells' low dimensional
        embedding, colored either by 2) the gene expression level or 3) the velocity magnitude values.
    """

    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    if background is not None:
        set_figure_params(background=background)

    mapper = get_mapper()

    point_size = (
        500.0 / np.sqrt(adata.shape[0])
        if pointsize is None
        else 500.0 / np.sqrt(adata.shape[0]) * pointsize
    )
    scatter_kwargs = dict(
        alpha=0.2, s=point_size, edgecolor=None, linewidth=0
    )  # (0, 0, 0, 1)
    if kwargs is not None:
        scatter_kwargs.update(kwargs)

    if type(genes) == str:
        genes = [genes]
    _genes = list(set(adata.var.index).intersection(genes))

    # avoid object for dtype in the gamma column https://stackoverflow.com/questions/40809503/python-numpy-typeerror-ufunc-isfinite-not-supported-for-the-input-types
    valid_id = np.isfinite(
        np.array(adata.var.loc[_genes, "gamma"], dtype="float")
    ).flatten()
    genes = np.array(_genes)[valid_id].tolist()
    # idx = [adata.var.index.to_list().index(i) for i in genes]

    if len(genes) == 0:
        raise Exception(
            "adata has no genes listed in your input gene vector or "
            "velocity estimation for those genes are not performed. "
            "Please try to run dyn.tl.dynamics(adata, filter_gene_mode='no')"
            "to estimate velocity for all genes: {}".format(_genes)
        )

    if not "X_" + basis in adata.obsm.keys():
        raise Exception("{} is not applied to adata.".format(basis))
    else:
        embedding = pd.DataFrame(
            {
                basis + "_0": adata.obsm["X_" + basis][:, x],
                basis + "_1": adata.obsm["X_" + basis][:, y],
            }
        )
        embedding.columns = ["dim_1", "dim_2"]

    if all([i in adata.layers.keys() for i in ["X_new", "X_total"]]) or all(
        [i in adata.layers.keys() for i in [mapper["X_new"], mapper["X_total"]]]
    ):
        mode = "labeling"
    elif all([i in adata.layers.keys() for i in ["X_spliced", "X_unspliced"]]) or all(
        [i in adata.layers.keys() for i in [mapper["X_spliced"], mapper["X_unspliced"]]]
    ):
        mode = "splicing"
    elif all(
        [i in adata.layers.keys() for i in ["X_uu", "X_ul", "X_su", "X_sl"]]
    ) or all(
        [
            i in adata.layers.keys()
            for i in [mapper["X_uu"], mapper["X_ul"], mapper["X_su"], mapper["X_sl"]]
        ]
    ):
        mode = "full"
    else:
        raise Exception(
            "your data should be in one of the proper mode: labelling (has X_new/X_total layers), splicing "
            "(has X_spliced/X_unspliced layers) or full (has X_uu/X_ul/X_su/X_sl layers)"
        )

    layers = list(adata.layers.keys())
    layers.extend(["X", "protein", "X_protein"])
    if ekey in layers:
        if ekey is "X":
            E_vec = (
                adata[:, genes].layers[mapper["X"]]
                if mapper["X"] in adata.layers.keys()
                else adata[:, genes].X
            )
        elif ekey in ["protein", "X_protein"]:
            E_vec = (
                adata[:, genes].layers[mapper[ekey]]
                if (ekey in mapper.keys()) and (mapper[ekey] in adata.obsm_keys())
                else adata[:, genes].obsm[ekey]
            )
        else:
            E_vec = (
                adata[:, genes].layers[mapper[ekey]]
                if (ekey in mapper.keys()) and (mapper[ekey] in adata.layers.keys())
                else adata[:, genes].layers[ekey]
            )

    n_cells, n_genes = adata.shape[0], len(genes)

    color_vec = np.repeat(np.nan, n_cells)
    if color is not None:
        color_vec = adata.obs[color].to_list()

    if "velocity_" not in vkey:
        vkey = "velocity_" + vkey
    if vkey == "velocity_U":
        V_vec = adata[:, genes].layers["velocity_U"]
        if "velocity_P" in adata.obsm.keys():
            P_vec = adata[:, genes].layer["velocity_P"]
    elif vkey == "velocity_S":
        V_vec = adata[:, genes].layers["velocity_S"]
        if "velocity_P" in adata.obsm.keys():
            P_vec = adata[:, genes].layers["velocity_P"]
    else:
        raise Exception(
            "adata has no vkey {} in either the layers or the obsm slot".format(vkey)
        )

    E_vec, V_vec = (
        E_vec.A if issparse(E_vec) else E_vec,
        V_vec.A if issparse(V_vec) else V_vec,
    )

    if "gamma" in adata.var.columns:
        if (
            not ("gamma_b" in adata.var.columns)
            or adata.var.gamma_b.unique()[0] is None
        ):
            adata.var.loc[:, "gamma_b"] = 0
        gamma, velocity_offset = (
            adata[:, genes].var.gamma.values,
            adata[:, genes].var.gamma_b.values,
        )
        (
            gamma[~np.isfinite(list(gamma))],
            velocity_offset[~np.isfinite(list(velocity_offset))],
        ) = (0, 0)
    else:
        raise Exception(
            "adata does not seem to have velocity_gamma column. Velocity estimation is required before "
            "running this function."
        )

    if mode is "labeling":
        new_mat, tot_mat = (
            adata[:, genes].layers[mapper["X_new"]],
            adata[:, genes].layers[mapper["X_total"]],
        )
        new_mat, tot_mat = (
            (new_mat.A, tot_mat.A) if issparse(new_mat) else (new_mat, tot_mat)
        )

        vel_u, vel_s = (
            adata[:, genes].layers["velocity_U"].A,
            adata[:, genes].layers["velocity_S"].A,
        )

        df = pd.DataFrame(
            {
                "new": new_mat.flatten(),
                "total": tot_mat.flatten(),
                "gene": genes * n_cells,
                "gamma": np.tile(gamma, n_cells),
                "velocity_offset": np.tile(velocity_offset, n_cells),
                "expression": E_vec.flatten(),
                "velocity": V_vec.flatten(),
                "color": np.repeat(color_vec, n_genes),
                "vel_u": vel_u.flatten(),
                "vel_s": vel_s.flatten(),
            },
            index=range(n_cells * n_genes),
        )

    elif mode is "splicing":
        unspliced_mat, spliced_mat = (
            adata[:, genes].layers[mapper["X_unspliced"]],
            adata[:, genes].layers[mapper["X_spliced"]],
        )
        unspliced_mat, spliced_mat = (
            (unspliced_mat.A, spliced_mat.A)
            if issparse(unspliced_mat)
            else (unspliced_mat, spliced_mat)
        )

        vel_u, vel_s = (
            np.zeros_like(adata[:, genes].layers["velocity_S"].A),
            adata[:, genes].layers["velocity_S"].A,
        )

        df = pd.DataFrame(
            {
                "unspliced": unspliced_mat.flatten(),
                "spliced": spliced_mat.flatten(),
                "gene": genes * n_cells,
                "gamma": np.tile(gamma, n_cells),
                "velocity_offset": np.tile(velocity_offset, n_cells),
                "expression": E_vec.flatten(),
                "velocity": V_vec.flatten(),
                "color": np.repeat(color_vec, n_genes),
                "vel_u": vel_u.flatten(),
                "vel_s": vel_s.flatten(),
            },
            index=range(n_cells * n_genes),
        )

    elif mode is "full":
        uu, ul, su, sl = (
            adata[:, genes].layers[mapper["X_uu"]],
            adata[:, genes].layers[mapper["X_ul"]],
            adata[:, genes].layers[mapper["X_su"]],
            adata[:, genes].layers[mapper["X_sl"]],
        )

        vel_u, vel_s = (
            np.zeros_like(adata[:, genes].layers["velocity_S"].A),
            adata[:, genes].layers["velocity_S"].A,
        )
        if "protein" in adata.obsm.keys():
            if "delta" in adata.var.columns:
                gamma_P = adata.var.delta[genes].values
                velocity_offset_P = (
                    [0] * n_cells
                    if (
                        not ("delta_b" in adata.var.columns)
                        or adata.var.gamma_b.unique() is None
                    )
                    else adata.var.delta_b[genes].values
                )
            else:
                raise Exception(
                    "adata does not seem to have velocity_gamma column. Velocity estimation is required before "
                    "running this function."
                )

            P = (
                adata[:, genes].obsm[mapper["X_protein"]]
                if (
                    ["X_protein"] in adata.obsm.keys()
                    or [mapper["X_protein"]] in adata.obsm.keys()
                )
                else adata[:, genes].obsm["protein"]
            )
            uu, ul, su, sl, P = (
                (uu.A, ul.A, su.A, sl.A, P.A) if issparse(uu) else (uu, ul, su, sl, P)
            )
            if issparse(P_vec):
                P_vec = P_vec.A

            vel_p = np.zeros_like(adata.obsm["velocity_P"][:, :])

            # df = pd.DataFrame({"uu": uu.flatten(), "ul": ul.flatten(), "su": su.flatten(), "sl": sl.flatten(), "P": P.flatten(),
            #                    'gene': genes * n_cells, 'prediction': np.tile(gamma, n_cells) * uu.flatten() +
            #                     np.tile(velocity_offset, n_cells), "velocity": genes * n_cells}, index=range(n_cells * n_genes))
            df = pd.DataFrame(
                {
                    "new": (ul + sl).flatten(),
                    "total": (uu + ul + sl + su).flatten(),
                    "S": (sl + su).flatten(),
                    "P": P.flatten(),
                    "gene": genes * n_cells,
                    "gamma": np.tile(gamma, n_cells),
                    "velocity_offset": np.tile(velocity_offset, n_cells),
                    "gamma_P": np.tile(gamma_P, n_cells),
                    "velocity_offset_P": np.tile(velocity_offset_P, n_cells),
                    "expression": E_vec.flatten(),
                    "velocity": V_vec.flatten(),
                    "velocity_protein": P_vec.flatten(),
                    "color": np.repeat(color_vec, n_genes),
                    "vel_u": vel_u.flatten(),
                    "vel_s": vel_s.flatten(),
                    "vel_p": vel_p.flatten(),
                },
                index=range(n_cells * n_genes),
            )
        else:
            vel_u, vel_s = (
                np.zeros_like(adata[:, genes].layers["velocity_S"].A),
                adata[:, genes].layers["velocity_S"].A,
            )
            df = pd.DataFrame(
                {
                    "new": (ul + sl).flatten(),
                    "total": (uu + ul + sl + su).flatten(),
                    "gene": genes * n_cells,
                    "gamma": np.tile(gamma, n_cells),
                    "velocity_offset": np.tile(velocity_offset, n_cells),
                    "expression": E_vec.flatten(),
                    "velocity": V_vec.flatten(),
                    "color": np.repeat(color_vec, n_genes),
                    "vel_u": vel_u.flatten(),
                    "vel_s": vel_s.flatten(),
                },
                index=range(n_cells * n_genes),
            )
    else:
        raise Exception(
            "Your adata is corrupted. Make sure that your layer has keys new, old for the labelling mode, "
            "spliced, ambiguous, unspliced for the splicing model and uu, ul, su, sl for the full mode"
        )

    num_per_gene = 6 if ("protein" in adata.obsm.keys() and mode is "full") else 3
    ncols = min([num_per_gene, ncols]) if ncols is not None else num_per_gene
    nrow, ncol = int(np.ceil(num_per_gene * n_genes / ncols)), ncols
    if figsize is None:
        g = plt.figure(None, (3 * ncol, 3 * nrow))  # , dpi=160
    else:
        g = plt.figure(None, (figsize[0] * ncol, figsize[1] * nrow))  # , dpi=160

    if rcParams.get("figure.facecolor") == "black":
        discrete_theme, continous_theme, divergent_theme = (
            "glasbey_dark",
            "inferno",
            "div_blue_black_red",
        )
    else:
        discrete_theme, continous_theme, divergent_theme = (
            "glasbey_white",
            "viridis",
            "div_blue_red",
        )

    discrete_cmap, discrete_color_key_cmap, discrete_background = (
        _themes[discrete_theme]["cmap"],
        _themes[discrete_theme]["color_key_cmap"],
        _themes[discrete_theme]["background"],
    )
    continous_cmap, continous_color_key_cmap, continous_background = (
        _themes[continous_theme]["cmap"],
        _themes[continous_theme]["color_key_cmap"],
        _themes[continous_theme]["background"],
    )
    divergent_cmap, divergent_color_key_cmap, divergent_background = (
        _themes[divergent_theme]["cmap"],
        _themes[divergent_theme]["color_key_cmap"],
        _themes[divergent_theme]["background"],
    )

    font_color = _select_font_color(discrete_background)

    # the following code is inspired by https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
    gs = plt.GridSpec(nrow, ncol)
    for i, gn in enumerate(genes):
        if num_per_gene is 3:
            ax1, ax2, ax3 = (
                plt.subplot(gs[i * 3]),
                plt.subplot(gs[i * 3 + 1]),
                plt.subplot(gs[i * 3 + 2]),
            )
        elif num_per_gene is 6:
            ax1, ax2, ax3, ax4, ax5, ax6 = (
                plt.subplot(gs[i * 3]),
                plt.subplot(gs[i * 3 + 1]),
                plt.subplot(gs[i * 3 + 2]),
                plt.subplot(gs[i * 3 + 3]),
                plt.subplot(gs[i * 3 + 4]),
                plt.subplot(gs[i * 3 + 5]),
            )
        try:
            ix = np.where(adata.var.index == gn)[0][0]
        except:
            continue
        cur_pd = df.loc[df.gene == gn, :]
        if cur_pd.color.isna().all():
            if cur_pd.shape[0] <= figsize[0] * figsize[1] * 1000000:
                ax1, color = _matplotlib_points(
                    cur_pd.iloc[:, [1, 0]].values,
                    ax=ax1,
                    labels=None,
                    values=cur_pd.loc[:, "expression"].values,
                    highlights=None,
                    cmap=continous_cmap,
                    color_key=None,
                    color_key_cmap=continous_color_key_cmap,
                    background=continous_background,
                    width=figsize[0],
                    height=figsize[1],
                    show_legend=legend,
                    **scatter_kwargs
                )
            else:
                ax1, color = _datashade_points(
                    cur_pd.iloc[:, [1, 0]].values,
                    ax=ax1,
                    labels=None,
                    values=cur_pd.loc[:, "expression"].values,
                    highlights=None,
                    cmap=continous_cmap,
                    color_key=None,
                    color_key_cmap=continous_color_key_cmap,
                    background=continous_background,
                    width=figsize[0],
                    height=figsize[1],
                    show_legend=legend,
                    **scatter_kwargs
                )
        else:
            if cur_pd.shape[0] <= figsize[0] * figsize[1] * 1000000:
                ax1, color = _matplotlib_points(
                    cur_pd.iloc[:, [1, 0]].values,
                    ax=ax1,
                    labels=cur_pd.loc[:, "color"],
                    values=None,
                    highlights=None,
                    cmap=discrete_cmap,
                    color_key=None,
                    color_key_cmap=discrete_color_key_cmap,
                    background=discrete_background,
                    width=figsize[0],
                    height=figsize[1],
                    show_legend=legend,
                    **scatter_kwargs
                )
            else:
                ax1, color = _datashade_points(
                    cur_pd.iloc[:, [1, 0]].values,
                    ax=ax1,
                    labels=cur_pd.loc[:, "color"],
                    values=None,
                    highlights=None,
                    cmap=discrete_cmap,
                    color_key=None,
                    color_key_cmap=discrete_color_key_cmap,
                    background=discrete_background,
                    width=figsize[0],
                    height=figsize[1],
                    show_legend=legend,
                    **scatter_kwargs
                )

        ax1.set_title(gn)
        ax1.set_xlabel("spliced")
        ax1.set_ylabel("unspliced")
        xnew = np.linspace(0, cur_pd.iloc[:, 1].max() * 0.80)
        ax1.plot(
            xnew,
            xnew * cur_pd.loc[:, "gamma"].unique()
            + cur_pd.loc[:, "velocity_offset"].unique(),
            c=font_color,
        )
        X_array, V_array = (
            cur_pd.iloc[:, [1, 0]].values,
            cur_pd.loc[:, ["vel_s", "vel_u"]].values,
        )

        # add quiver:
        if show_quiver:
            V_array /= 3 * quiver_autoscaler(X_array, V_array)

            if background is None:
                background = rcParams.get("figure.facecolor")
            if quiver_size is None:
                quiver_size = 1
            if background == "black":
                edgecolors = "white"
            else:
                edgecolors = "black"

            head_w, head_l, ax_l, scale = default_quiver_args(
                quiver_size, quiver_length
            )

            quiver_kwargs = {
                "angles": "xy",
                "scale": scale,
                "scale_units": "xy",
                "width": 0.0005,
                "headwidth": head_w,
                "headlength": head_l,
                "headaxislength": ax_l,
                "minshaft": 1,
                "minlength": 1,
                "pivot": "tail",
                "linewidth": 0.2,
                "edgecolors": edgecolors,
                "linewidth": 0.2,
                "color": color,
                "alpha": 1,
                "zorder": 10,
            }
            quiver_kwargs = update_dict(quiver_kwargs, q_kwargs_dict)
            ax1.quiver(
                X_array[:, 0],
                X_array[:, 1],
                V_array[:, 0],
                V_array[:, 1],
                **quiver_kwargs
            )

        ax1.set_xlim(0, np.max(X_array[:, 0]) * 1.02)
        ax1.set_ylim(0, np.max(X_array[:, 1]) * 1.02)

        despline(ax1)  # sns.despline()

        df_embedding = pd.concat([cur_pd, embedding], axis=1)
        V_vec = cur_pd.loc[:, "velocity"]

        limit = np.nanmax(
            np.abs(np.nanpercentile(V_vec, [1, 99]))
        )  # upper and lowe limit / saturation

        V_vec = V_vec + limit  # that is: tmp_colorandum - (-limit)
        V_vec = V_vec / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
        V_vec = np.clip(V_vec, 0, 1)

        if cur_pd.shape[0] <= figsize[0] * figsize[1] * 1000000:
            ax2, _ = _matplotlib_points(
                embedding.iloc[:, :2].values,
                ax=ax2,
                labels=None,
                values=cur_pd.loc[:, "expression"].values,
                highlights=None,
                cmap=continous_cmap,
                color_key=None,
                color_key_cmap=continous_color_key_cmap,
                background=continous_background,
                width=figsize[0],
                height=figsize[1],
                show_legend=legend,
                **scatter_kwargs
            )
        else:
            ax2, _ = _datashade_points(
                embedding.iloc[:, :2].values,
                ax=ax2,
                labels=None,
                values=cur_pd.loc[:, "expression"].values,
                highlights=None,
                cmap=continous_cmap,
                color_key=None,
                color_key_cmap=continous_color_key_cmap,
                background=continous_background,
                width=figsize[0],
                height=figsize[1],
                show_legend=legend,
                **scatter_kwargs
            )

        ax2.set_title(gn + " (" + ekey + ")")
        ax2.set_xlabel(basis + "_1")
        ax2.set_ylabel(basis + "_2")

        if cur_pd.shape[0] <= figsize[0] * figsize[1] * 1000000:
            ax3, _ = _matplotlib_points(
                embedding.iloc[:, :2].values,
                ax=ax3,
                labels=None,
                values=V_vec.values,
                highlights=None,
                cmap=divergent_cmap,
                color_key=None,
                color_key_cmap=divergent_color_key_cmap,
                background=divergent_background,
                width=figsize[0],
                height=figsize[1],
                show_legend=legend,
                **scatter_kwargs
            )
        else:
            ax3, _ = _datashade_points(
                embedding.iloc[:, :2].values,
                ax=ax3,
                labels=None,
                values=V_vec.values,
                highlights=None,
                cmap=divergent_cmap,
                color_key=None,
                color_key_cmap=divergent_color_key_cmap,
                background=divergent_background,
                width=figsize[0],
                height=figsize[1],
                show_legend=legend,
                **scatter_kwargs
            )

        ax3.set_title(gn + " (" + vkey + ")")
        ax3.set_xlabel(basis + "_1")
        ax3.set_ylabel(basis + "_2")

        if (
            "protein" in adata.obsm.keys()
            and mode is "full"
            and all([i in adata.layers.keys() for i in ["uu", "ul", "su", "sl"]])
        ):
            if cur_pd.color.unique() != np.nan:
                if cur_pd.shape[0] <= figsize[0] * figsize[1] * 1000000:
                    ax4, color = _matplotlib_points(
                        cur_pd.iloc[:, [3, 2]].values,
                        ax=ax4,
                        labels=None,
                        values=cur_pd.loc[:, "expression"].values,
                        highlights=None,
                        cmap=continous_cmap,
                        color_key=None,
                        color_key_cmap=continous_color_key_cmap,
                        background=continous_background,
                        width=figsize[0],
                        height=figsize[1],
                        show_legend=legend,
                        **scatter_kwargs
                    )
                else:
                    ax4, color = _datashade_points(
                        cur_pd.iloc[:, [3, 2]].values,
                        ax=ax4,
                        labels=None,
                        values=cur_pd.loc[:, "expression"].values,
                        highlights=None,
                        cmap=continous_cmap,
                        color_key=None,
                        color_key_cmap=continous_color_key_cmap,
                        background=continous_background,
                        width=figsize[0],
                        height=figsize[1],
                        show_legend=legend,
                        **scatter_kwargs
                    )
            else:
                if cur_pd.shape[0] <= figsize[0] * figsize[1] * 1000000:
                    ax4, color = _matplotlib_points(
                        cur_pd.iloc[:, [1, 0]].values,
                        ax=ax4,
                        labels=color,
                        values=None,
                        highlights=None,
                        cmap=discrete_cmap,
                        color_key=None,
                        color_key_cmap=discrete_color_key_cmap,
                        background=discrete_background,
                        width=figsize[0],
                        height=figsize[1],
                        show_legend=legend,
                        **scatter_kwargs
                    )
                else:
                    ax4, color = _datashade_points(
                        cur_pd.iloc[:, [1, 0]].values,
                        ax=ax4,
                        labels=color,
                        values=None,
                        highlights=None,
                        cmap=discrete_cmap,
                        color_key=None,
                        color_key_cmap=discrete_color_key_cmap,
                        background=discrete_background,
                        width=figsize[0],
                        height=figsize[1],
                        show_legend=legend,
                        **scatter_kwargs
                    )

            ax4.set_title(gn)
            ax1.set_xlabel("spliced")
            ax1.set_ylabel("protein")

            xnew = np.linspace(0, cur_pd.iloc[:, 3].max())
            ax4.plot(
                xnew,
                xnew * cur_pd.loc[:, "gamma_P"].unique()
                + cur_pd.loc[:, "velocity_offset_P"].unique(),
                c=font_color,
            )
            X_array, V_array = (
                cur_pd.iloc[:, [3, 2]].values,
                cur_pd.loc[:, ["vel_p", "vel_s"]].values,
            )

            # add quiver:
            if show_quiver:
                V_array /= 3 * quiver_autoscaler(X_array, V_array)

                if background is None:
                    background = rcParams.get("figure.facecolor")
                if quiver_size is None:
                    quiver_size = 1
                if background == "black":
                    edgecolors = "white"
                else:
                    edgecolors = "black"

                head_w, head_l, ax_l, scale = default_quiver_args(
                    quiver_size, quiver_length
                )

                quiver_kwargs = {
                    "angles": "xy",
                    "scale": scale,
                    "scale_units": "xy",
                    "width": 0.0005,
                    "headwidth": head_w,
                    "headlength": head_l,
                    "headaxislength": ax_l,
                    "minshaft": 1,
                    "minlength": 1,
                    "pivot": "tail",
                    "linewidth": 0.2,
                    "edgecolors": edgecolors,
                    "linewidth": 0.2,
                    "color": color,
                    "alpha": 1,
                    "zorder": 10,
                }
                quiver_kwargs = update_dict(quiver_kwargs, q_kwargs_dict)
                ax4.quiver(
                    X_array[:, 0],
                    X_array[:, 1],
                    V_array[:, 0],
                    V_array[:, 1],
                    **quiver_kwargs
                )

            ax4.set_ylim(0, np.max(X_array[:, 0]) * 1.02)
            ax4.set_xlim(0, np.max(X_array[:, 1]) * 1.02)

            despline(ax1)  # sns.despline()

            V_vec = df_embedding.loc[:, "velocity_p"]

            limit = np.nanmax(
                np.abs(np.nanpercentile(V_vec, [1, 99]))
            )  # upper and lowe limit / saturation

            V_vec = V_vec + limit  # that is: tmp_colorandum - (-limit)
            V_vec = V_vec / (2 * limit)  # that is: tmp_colorandum / (limit - (-limit))
            V_vec = np.clip(V_vec, 0, 1)

            if cur_pd.shape[0] <= figsize[0] * figsize[1] * 1000000:
                ax5, _ = _matplotlib_points(
                    embedding.iloc[:, :2],
                    ax=ax5,
                    labels=None,
                    values=embedding.loc[:, "P"].values,
                    highlights=None,
                    cmap=continous_cmap,
                    color_key=None,
                    color_key_cmap=continous_color_key_cmap,
                    background=continous_background,
                    width=figsize[0],
                    height=figsize[1],
                    show_legend=legend,
                    **scatter_kwargs
                )
            else:
                ax5, _ = _datashade_points(
                    embedding.iloc[:, :2],
                    ax=ax5,
                    labels=None,
                    values=embedding.loc[:, "P"].values,
                    highlights=None,
                    cmap=continous_cmap,
                    color_key=None,
                    color_key_cmap=continous_color_key_cmap,
                    background=continous_background,
                    width=figsize[0],
                    height=figsize[1],
                    show_legend=legend,
                    **scatter_kwargs
                )

            ax5.set_title(gn + " (protein expression)")
            ax5.set_xlabel(basis + "_1")
            ax5.set_ylabel(basis + "_2")

            if cur_pd.shape[0] <= figsize[0] * figsize[1] * 1000000:
                ax6, _ = _matplotlib_points(
                    embedding.iloc[:, :2],
                    ax=ax6,
                    labels=None,
                    values=V_vec.values,
                    highlights=None,
                    cmap=divergent_cmap,
                    color_key=None,
                    color_key_cmap=divergent_color_key_cmap,
                    background=divergent_background,
                    width=figsize[0],
                    height=figsize[1],
                    show_legend=legend,
                    **scatter_kwargs
                )
            else:
                ax6, _ = _datashade_points(
                    embedding.iloc[:, :2],
                    ax=ax6,
                    labels=None,
                    values=V_vec.values,
                    highlights=None,
                    cmap=divergent_cmap,
                    color_key=None,
                    color_key_cmap=divergent_color_key_cmap,
                    background=divergent_background,
                    width=figsize[0],
                    height=figsize[1],
                    show_legend=legend,
                    **scatter_kwargs
                )

            ax6.set_title(gn + " (protein velocity)")
            ax6.set_xlabel(basis + "_1")
            ax6.set_ylabel(basis + "_2")

    plt.tight_layout()
    plt.show()
    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'phase_portraits', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return g


def dynamics(
    adata,
    vkey,
    unit="hours",
    log_unnormalized=True,
    y_log_scale=False,
    group=None,
    ncols=None,
    figsize=None,
    dpi=None,
    boxwidth=None,
    barwidth=None,
    true_param_prefix=None,
    save_show_or_return='show',
    save_kwargs={},
):
    """Plot the data and fitting of different metabolic labeling experiments.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        vkey: list of `str`
            key for variable or gene names.
        unit: `str` (default: `hour`)
            The unit of the labeling time, for example, `hours` or `minutes`.
        y_log_scale: `bool` (default: `True`)
            Whether or not to use log scale for y-axis.
        group: `str` or None (default: `None`)
            The key for the group annotation in .obs attribute. Currently not used.
        ncols: `int` or None (default: `None`)
            The number of columns in the plot.
        figsize: `[float, float]` or `(float, float)` or None
            The size of figure.
        dpi: `float` or None
            Figure resolution.
        boxwidth: `float`
            The width of the box of the boxplot.
        barwidth: `float`
            The width of the bar of the barplot.
        true_param_prefix: `str`
            The prefix for the column names of true parameters in the .var attributes. Useful for the simulation data.
        save_show_or_return: {'show', 'save', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save function. By default it is an empty dictionary and the save function
            will use the {"path": None, "prefix": 'dynamics', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.

    Returns
    -------
        Nothing but plot the figure of the metabolic fitting.
    """

    import matplotlib.pyplot as plt

    if group is not None and group + "_dynamics" in adata.uns_keys():
        uns_key = group + "_dynamics"
        _group, grp_len = np.unique(adata.obs[group]), len(np.unique(adata.obs[group]))
    else:
        uns_key = "dynamics"
        _group, grp_len = ["_all_cells"], 1

    (
        T,
        group,
        asspt_mRNA,
        experiment_type,
        _,
        mode,
        has_splicing,
        has_labeling,
        has_protein,
    ) = adata.uns[uns_key].values()
    if asspt_mRNA is "ss":
        # run the phase plot
        warnings.warn(
            "dynamics plot doesn't support steady state mode, use phase_portraits function instead."
        )
        phase_portraits(adata)

    T_uniq = np.unique(T)
    t = np.linspace(0, T_uniq[-1], 1000)
    gene_idx = [np.where(adata.var.index.values == gene)[0][0] for gene in vkey]

    if boxwidth is None and len(T_uniq) > 1:
        boxwidth = 0.8 * (np.diff(T_uniq).min() / 2)

    if barwidth is None and len(T_uniq) > 1:
        barwidth = 0.8 * (np.diff(T_uniq).min() / 2)

    if has_splicing:
        if mode is "moment":
            sub_plot_n = 4  # number of subplots for each gene
        elif experiment_type is "kin" or experiment_type is "deg":
            sub_plot_n = 4
        elif experiment_type is "one_shot":  # just the labeled RNA
            sub_plot_n = 1
        elif experiment_type is "mix_std_stm":
            sub_plot_n = 5
    else:
        if mode is "moment":
            sub_plot_n = 2
        elif experiment_type is "kin" or experiment_type is "deg":
            sub_plot_n = 2
        elif experiment_type is "one_shot":  # just the labeled RNA
            sub_plot_n = 1
        elif experiment_type is "mix_std_stm":
            sub_plot_n = 3

    ncols = (
        len(gene_idx) * grp_len
        if ncols is None
        else min(len(gene_idx) * grp_len, ncols)
    )
    nrows = int(np.ceil(len(gene_idx) * sub_plot_n * grp_len / ncols))
    figsize = [7, 5] if figsize is None else figsize
    g = plt.figure(None, (figsize[0] * ncols, figsize[1] * nrows), dpi=dpi)
    gs = plt.GridSpec(
        nrows,
        ncols,
        g,
    )

    # we need to visualize gene in row-wise mode
    for grp_idx, cur_grp in enumerate(_group):
        if cur_grp == "_all_cells":
            prefix = ""  # kinetic_parameter_
        else:
            prefix = group + "_" + cur_grp + "_"

        for i, idx in enumerate(gene_idx):
            gene_name = adata.var_names[idx]

            if mode is "moment":
                a, b, alpha_a, alpha_i, beta, gamma = adata.var.loc[
                    gene_name,
                    [
                        prefix + "a",
                        prefix + "b",
                        prefix + "alpha_a",
                        prefix + "alpha_i",
                        prefix + "beta",
                        prefix + "gamma",
                    ],
                ]
                params = {
                    "a": a,
                    "b": b,
                    "alpha_a": alpha_a,
                    "alpha_i": alpha_i,
                    "beta": beta,
                    "gamma": gamma,
                }  # "la": 1, "si": 0,
                mom = moments(*list(params.values()))
                mom.integrate(t)
                mom_data = (
                    mom.get_all_central_moments()
                    if has_splicing
                    else mom.get_nosplice_central_moments()
                )
                if true_param_prefix is not None:
                    (
                        true_a,
                        true_b,
                        true_alpha_a,
                        true_alpha_i,
                        true_beta,
                        true_gamma,
                    ) = (
                        adata.var.loc[gene_name, true_param_prefix + "a"]
                        if true_param_prefix + "a" in adata.var_keys()
                        else -np.inf,
                        adata.var.loc[gene_name, true_param_prefix + "b"]
                        if true_param_prefix + "b" in adata.var_keys()
                        else -np.inf,
                        adata.var.loc[gene_name, true_param_prefix + "alpha_a"]
                        if true_param_prefix + "alpha_a" in adata.var_keys()
                        else -np.inf,
                        adata.var.loc[gene_name, true_param_prefix + "alpha_i"]
                        if true_param_prefix + "alpha_i" in adata.var_keys()
                        else -np.inf,
                        adata.var.loc[gene_name, true_param_prefix + "beta"]
                        if true_param_prefix + "beta" in adata.var_keys()
                        else -np.inf,
                        adata.var.loc[gene_name, true_param_prefix + "gamma"]
                        if true_param_prefix + "gamma" in adata.var_keys()
                        else -np.inf,
                    )

                    true_params = {
                        "a": true_a,
                        "b": true_b,
                        "alpha_a": true_alpha_a,
                        "alpha_i": true_alpha_i,
                        "beta": true_beta,
                        "gamma": true_gamma,
                    }  # "la": 1, "si": 0,
                    true_mom = moments(*list(true_params.values()))
                    true_mom.integrate(t)
                    true_mom_data = (
                        true_mom.get_all_central_moments()
                        if has_splicing
                        else true_mom.get_nosplice_central_moments()
                    )

                # n_mean, n_var = x_data[:2, :], x_data[2:, :]
                if has_splicing:
                    tmp = (
                        [
                            adata[:, gene_idx].layers["X_ul"].A.T,
                            adata.layers["X_sl"].A.T,
                        ]
                        if "X_ul" in adata.layers.keys()
                        else [
                            adata[:, gene_idx].layers["ul"].A.T,
                            adata.layers["sl"].A.T,
                        ]
                    )
                    x_data = [tmp[0].A, tmp[1].A] if issparse(tmp[0]) else tmp
                    if log_unnormalized and "X_ul" not in adata.layers.keys():
                        x_data = [np.log(tmp[0] + 1), np.log(tmp[1] + 1)]

                    title_ = [
                        "(unspliced labeled)",
                        "(spliced labeled)",
                        "(unspliced labeled)",
                        "(spliced labeled)",
                    ]
                    Obs_m = [adata.uns["M_ul"], adata.uns["M_sl"]]
                    Obs_v = [adata.uns["V_ul"], adata.uns["V_sl"]]
                    j_species = 2  # number of species
                else:
                    tmp = (
                        adata[:, gene_idx].layers["X_new"].T
                        if "X_new" in adata.layers.keys()
                        else adata[:, gene_idx].layers["new"].T
                    )
                    x_data = [tmp.A] if issparse(tmp) else [tmp]

                    if log_unnormalized and "X_new" not in adata.layers.keys():
                        x_data = [np.log(x_data[0] + 1)]
                    # only use new key for calculation, so we only have M, V
                    title_ = [" (labeled)", " (labeled)"]
                    Obs_m, Obs_v = [adata.uns["M"]], [adata.uns["V"]]
                    j_species = 1

                for j in range(sub_plot_n):
                    row_ind = int(
                        np.floor(i / ncols)
                    )  # make sure all related plots for the same gene in the same column.
                    ax = plt.subplot(
                        gs[
                            (row_ind * sub_plot_n + j) * ncols * grp_len
                            + (i % ncols - 1) * grp_len
                            + 1
                        ]
                    )
                    if true_param_prefix is not None and j == 0:
                        if has_splicing:
                            ax.text(
                                0.05,
                                0.92,
                                r"$a$"
                                + ": {0:.2f}; ".format(true_a)
                                + r"$\hat a$"
                                + ": {0:.2f} \n".format(a)
                                + r"$b$"
                                + ": {0:.2f}; ".format(true_b)
                                + r"$\hat b$"
                                + ": {0:.2f} \n".format(b)
                                + r"$\alpha_a$"
                                + ": {0:.2f}; ".format(true_alpha_a)
                                + r"$\hat \alpha_a$"
                                + ": {0:.2f} \n".format(alpha_a)
                                + r"$\alpha_i$"
                                + ": {0:.2f}; ".format(true_alpha_i)
                                + r"$\hat \alpha_i$"
                                + ": {0:.2f} \n".format(alpha_i)
                                + r"$\beta$"
                                + ": {0:.2f}; ".format(true_beta)
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta)
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma)
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.05,
                                0.92,
                                r"$a$"
                                + ": {0:.2f}; ".format(true_a)
                                + r"$\hat a$"
                                + ": {0:.2f} \n".format(a)
                                + r"$b$"
                                + ": {0:.2f}; ".format(true_b)
                                + r"$\hat b$"
                                + ": {0:.2f} \n".format(b)
                                + r"$\alpha_a$"
                                + ": {0:.2f}; ".format(true_alpha_a)
                                + r"$\hat \alpha_a$"
                                + ": {0:.2f} \n".format(alpha_a)
                                + r"$\alpha_i$"
                                + ": {0:.2f}; ".format(true_alpha_i)
                                + r"$\hat \alpha_i$"
                                + ": {0:.2f} \n".format(alpha_i)
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma)
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma),
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                            )
                    if j < j_species:
                        ax.boxplot(
                            x=[x_data[j][i][T == std] for std in T_uniq],
                            positions=T_uniq,
                            widths=boxwidth,
                            showfliers=False,
                            showmeans=True,
                        )  # x=T.values, y= # ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
                        # ax.scatter(T_uniq, Obs_m[j][i], c='r')  # ax1.plot(T, u.T, linestyle='None', marker='o', markersize=10)
                        if y_log_scale:
                            ax.set_yscale("log")
                        if log_unnormalized:
                            ax.set_ylabel("Expression (log)")  #
                        else:
                            ax.set_ylabel("Expression")
                        ax.plot(t, mom_data[j], "k--")
                        if true_param_prefix is not None:
                            ax.plot(t, true_mom_data[j], "r--")
                    else:
                        ax.scatter(T_uniq, Obs_v[j - j_species][i])  # , c='r'
                        if y_log_scale:
                            ax.set_yscale("log")
                        if log_unnormalized:
                            ax.set_ylabel("Variance (log expression)")  #
                        else:
                            ax.set_ylabel("Variance")
                        ax.plot(t, mom_data[j], "k--")
                        if true_param_prefix is not None:
                            ax.plot(t, true_mom_data[j], "r--")

                    ax.set_xlabel("time (" + unit + ")")
                    ax.set_title(gene_name + " " + title_[j])

            elif experiment_type is "deg":
                if has_splicing:
                    layers = (
                        ["X_uu", "X_ul", "X_su", "X_sl"]
                        if "X_ul" in adata.layers.keys()
                        else ["uu", "ul", "su", "sl"]
                    )
                    uu, ul, su, sl = (
                        adata[:, gene_name].layers[layers[0]],
                        adata[:, gene_name].layers[layers[1]],
                        adata[:, gene_name].layers[layers[2]],
                        adata[:, gene_name].layers[layers[3]],
                    )
                    uu, ul, su, sl = (
                        (
                            uu.toarray().squeeze(),
                            ul.toarray().squeeze(),
                            su.toarray().squeeze(),
                            sl.toarray().squeeze(),
                        )
                        if issparse(uu)
                        else (uu.squeeze(), ul.squeeze(), su.squeeze(), sl.squeeze())
                    )

                    if log_unnormalized and layers == ["uu", "ul", "su", "sl"]:
                        uu, ul, su, sl = (
                            np.log(uu + 1),
                            np.log(ul + 1),
                            np.log(su + 1),
                            np.log(sl + 1),
                        )

                    alpha, beta, gamma, ul0, sl0, uu0, half_life = adata.var.loc[
                        gene_name,
                        [
                            prefix + "alpha",
                            prefix + "beta",
                            prefix + "gamma",
                            prefix + "ul0",
                            prefix + "sl0",
                            prefix + "uu0",
                            "half_life",
                        ],
                    ]
                    # $u$ - unlabeled, unspliced
                    # $s$ - unlabeled, spliced
                    # $w$ - labeled, unspliced
                    # $l$ - labeled, spliced
                    #
                    beta = sys.float_info.epsilon if beta == 0 else beta
                    gamma = sys.float_info.epsilon if gamma == 0 else gamma
                    u = sol_u(t, uu0, alpha, beta)
                    su0 = np.mean(su[T == np.min(T)])  # this should also be estimated
                    s = sol_s(t, su0, uu0, alpha, beta, gamma)
                    w = sol_u(t, ul0, 0, beta)
                    l = sol_s(t, sl0, ul0, 0, beta, gamma)
                    if true_param_prefix is not None:
                        true_alpha, true_beta, true_gamma = (
                            adata.var.loc[gene_name, true_param_prefix + "alpha"]
                            if true_param_prefix + "alpha" in adata.var_keys()
                            else -np.inf,
                            adata.var.loc[gene_name, true_param_prefix + "beta"]
                            if true_param_prefix + "beta" in adata.var_keys()
                            else -np.inf,
                            adata.var.loc[gene_name, true_param_prefix + "gamma"]
                            if true_param_prefix + "gamma" in adata.var_keys()
                            else -np.inf,
                        )

                        true_u = sol_u(t, uu0, true_alpha, true_beta)
                        true_s = sol_s(t, su0, uu0, true_alpha, true_beta, true_gamma)
                        true_w = sol_u(t, ul0, 0, true_beta)
                        true_l = sol_s(t, sl0, ul0, 0, true_beta, true_gamma)

                        true_p = np.vstack((true_u, true_w, true_s, true_l))

                    title_ = [
                        "(unspliced unlabeled)",
                        "(unspliced labeled)",
                        "(spliced unlabeled)",
                        "(spliced labeled)",
                    ]

                    Obs, Pred = np.vstack((uu, ul, su, sl)), np.vstack((u, w, s, l))
                else:
                    layers = (
                        ["X_new", "X_total"]
                        if "X_new" in adata.layers.keys()
                        else ["new", "total"]
                    )
                    uu, ul = (
                        adata[:, gene_name].layers[layers[1]]
                        - adata[:, gene_name].layers[layers[0]],
                        adata[:, gene_name].layers[layers[0]],
                    )
                    uu, ul = (
                        (uu.toarray().squeeze(), ul.toarray().squeeze())
                        if issparse(uu)
                        else (uu.squeeze(), ul.squeeze())
                    )

                    if log_unnormalized and layers == ["new", "total"]:
                        uu, ul = np.log(uu + 1), np.log(ul + 1)

                    alpha, gamma, uu0, ul0, half_life = adata.var.loc[
                        gene_name,
                        [
                            prefix + "alpha",
                            prefix + "gamma",
                            prefix + "uu0",
                            prefix + "ul0",
                            "half_life",
                        ],
                    ]

                    # require no beta functions
                    u = sol_u(t, uu0, alpha, gamma)
                    s = None  # sol_s(t, su0, uu0, 0, 1, gamma)
                    w = sol_u(t, ul0, 0, gamma)
                    l = None  # sol_s(t, 0, 0, alpha, 1, gamma)
                    title_ = ["(unlabeled)", "(labeled)"]
                    if true_param_prefix is not None:
                        true_alpha, true_gamma = (
                            adata.var.loc[gene_name, true_param_prefix + "alpha"]
                            if true_param_prefix + "alpha" in adata.var_keys()
                            else -np.inf,
                            adata.var.loc[gene_name, true_param_prefix + "gamma"]
                            if true_param_prefix + "gamma" in adata.var_keys()
                            else -np.inf,
                        )
                        true_u = sol_u(t, uu0, true_alpha, true_gamma)
                        true_w = sol_u(t, ul0, 0, true_gamma)

                        true_p = np.vstack((true_u, true_w))

                    Obs, Pred = np.vstack((uu, ul)), np.vstack((u, w))

                for j in range(sub_plot_n):
                    row_ind = int(
                        np.floor(i / ncols)
                    )  # make sure unlabled and labeled are in the same column.
                    ax = plt.subplot(
                        gs[
                            (row_ind * sub_plot_n + j) * ncols * grp_len
                            + (i % ncols - 1) * grp_len
                            + 1
                        ]
                    )
                    if true_param_prefix is not None and j == 0:
                        if has_splicing:
                            ax.text(
                                0.75,
                                0.50,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha)
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha)
                                + r"$\beta$"
                                + ": {0:.2f}; ".format(true_beta)
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta)
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma)
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma),
                                ha="right",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.75,
                                0.50,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha)
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha)
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma)
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma),
                                ha="right",
                                va="top",
                                transform=ax.transAxes,
                            )

                    ax.boxplot(
                        x=[Obs[j][T == std] for std in T_uniq],
                        positions=T_uniq,
                        widths=boxwidth,
                        showfliers=False,
                        showmeans=True,
                    )
                    ax.plot(t, Pred[j], "k--")
                    if true_param_prefix is not None:
                        ax.plot(t, true_p[j], "r--")
                    if j == sub_plot_n - 1:
                        ax.text(
                            0.8,
                            0.8,
                            r"$t_{1/2} = $" + "{0:.2f}".format(half_life) + unit[0],
                            ha="right",
                            va="top",
                            transform=ax.transAxes,
                        )
                    ax.set_xlabel("time (" + unit + ")")
                    if y_log_scale:
                        ax.set_yscale("log")
                    if log_unnormalized:
                        ax.set_ylabel("Expression (log)")
                    else:
                        ax.set_ylabel("Expression")
                    ax.set_title(gene_name + " " + title_[j])
            elif experiment_type is "kin":
                if has_splicing:
                    layers = (
                        ["X_uu", "X_ul", "X_su", "X_sl"]
                        if "X_ul" in adata.layers.keys()
                        else ["uu", "ul", "su", "sl"]
                    )
                    uu, ul, su, sl = (
                        adata[:, gene_name].layers[layers[0]],
                        adata[:, gene_name].layers[layers[1]],
                        adata[:, gene_name].layers[layers[2]],
                        adata[:, gene_name].layers[layers[3]],
                    )
                    uu, ul, su, sl = (
                        (
                            uu.toarray().squeeze(),
                            ul.toarray().squeeze(),
                            su.toarray().squeeze(),
                            sl.toarray().squeeze(),
                        )
                        if issparse(uu)
                        else (uu.squeeze(), ul.squeeze(), su.squeeze(), sl.squeeze())
                    )

                    if log_unnormalized and layers == ["uu", "ul", "su", "sl"]:
                        uu, ul, su, sl = (
                            np.log(uu + 1),
                            np.log(ul + 1),
                            np.log(su + 1),
                            np.log(sl + 1),
                        )

                    alpha, beta, gamma, uu0, su0 = adata.var.loc[
                        gene_name,
                        [
                            prefix + "alpha",
                            prefix + "beta",
                            prefix + "gamma",
                            prefix + "uu0",
                            prefix + "su0",
                        ],
                    ]
                    # $u$ - unlabeled, unspliced
                    # $s$ - unlabeled, spliced
                    # $w$ - labeled, unspliced
                    # $l$ - labeled, spliced
                    #
                    beta = sys.float_info.epsilon if beta == 0 else beta
                    gamma = sys.float_info.epsilon if gamma == 0 else gamma
                    u = sol_u(t, uu0, 0, beta)
                    s = sol_s(t, su0, uu0, 0, beta, gamma)
                    w = sol_u(t, 0, alpha, beta)
                    l = sol_s(t, 0, 0, alpha, beta, gamma)
                    if true_param_prefix is not None:
                        true_alpha, true_beta, true_gamma = (
                            adata.var.loc[gene_name, true_param_prefix + "alpha"]
                            if true_param_prefix + "alpha" in adata.var_keys()
                            else -np.inf,
                            adata.var.loc[gene_name, true_param_prefix + "beta"]
                            if true_param_prefix + "beta" in adata.var_keys()
                            else -np.inf,
                            adata.var.loc[gene_name, true_param_prefix + "gamma"]
                            if true_param_prefix + "gamma" in adata.var_keys()
                            else -np.inf,
                        )
                        true_u = sol_u(t, uu0, 0, true_beta)
                        true_s = sol_s(t, su0, uu0, 0, true_beta, true_gamma)
                        true_w = sol_u(t, 0, true_alpha, true_beta)
                        true_l = sol_s(t, 0, 0, true_alpha, true_beta, true_gamma)

                        true_p = np.vstack((true_u, true_w, true_s, true_l))

                    title_ = [
                        "(unspliced unlabeled)",
                        "(unspliced labeled)",
                        "(spliced unlabeled)",
                        "(spliced labeled)",
                    ]

                    Obs, Pred = np.vstack((uu, ul, su, sl)), np.vstack((u, w, s, l))
                else:
                    layers = (
                        ["X_new", "X_total"]
                        if "X_new" in adata.layers.keys()
                        else ["new", "total"]
                    )
                    uu, ul = (
                        adata[:, gene_name].layers[layers[1]]
                        - adata[:, gene_name].layers[layers[0]],
                        adata[:, gene_name].layers[layers[0]],
                    )
                    uu, ul = (
                        (uu.toarray().squeeze(), ul.toarray().squeeze())
                        if issparse(uu)
                        else (uu.squeeze(), ul.squeeze())
                    )

                    if log_unnormalized and layers == ["new", "total"]:
                        uu, ul = np.log(uu + 1), np.log(ul + 1)

                    alpha, gamma, uu0 = adata.var.loc[
                        gene_name, [prefix + "alpha", prefix + "gamma", prefix + "uu0"]
                    ]

                    # require no beta functions
                    u = sol_u(t, uu0, 0, gamma)
                    s = None  # sol_s(t, su0, uu0, 0, 1, gamma)
                    w = sol_u(t, 0, alpha, gamma)
                    l = None  # sol_s(t, 0, 0, alpha, 1, gamma)
                    if true_param_prefix is not None:
                        true_alpha, true_gamma = (
                            adata.var.loc[gene_name, true_param_prefix + "alpha"]
                            if true_param_prefix + "alpha" in adata.var_keys()
                            else -np.inf,
                            adata.var.loc[gene_name, true_param_prefix + "gamma"]
                            if true_param_prefix + "gamma" in adata.var_keys()
                            else -np.inf,
                        )
                        true_u = sol_u(t, uu0, 0, true_gamma)
                        true_w = sol_u(t, 0, true_alpha, true_gamma)

                        true_p = np.vstack((true_u, true_w))

                    title_ = ["(unlabeled)", "(labeled)"]

                    Obs, Pred = np.vstack((uu, ul)), np.vstack((u, w))

                for j in range(sub_plot_n):
                    row_ind = int(
                        np.floor(i / ncols)
                    )  # make sure unlabled and labeled are in the same column.
                    ax = plt.subplot(
                        gs[
                            (row_ind * sub_plot_n + j) * ncols * grp_len
                            + (i % ncols - 1) * grp_len
                            + 1
                        ]
                    )
                    if true_param_prefix is not None and j == 0:
                        if has_splicing:
                            ax.text(
                                0.75,
                                0.90,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha)
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha)
                                + r"$\beta$"
                                + ": {0:.2f}; ".format(true_beta)
                                + r"$\hat \beta$"
                                + ": {0:.2f} \n".format(beta)
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma)
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma),
                                ha="right",
                                va="top",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.text(
                                0.75,
                                0.90,
                                r"$\alpha$"
                                + ": {0:.2f}; ".format(true_alpha)
                                + r"$\hat \alpha$"
                                + ": {0:.2f} \n".format(alpha)
                                + r"$\gamma$"
                                + ": {0:.2f}; ".format(true_gamma)
                                + r"$\hat \gamma$"
                                + ": {0:.2f} \n".format(gamma),
                                ha="right",
                                va="top",
                                transform=ax.transAxes,
                            )

                    ax.boxplot(
                        x=[Obs[j][T == std] for std in T_uniq],
                        positions=T_uniq,
                        widths=boxwidth,
                        showfliers=False,
                        showmeans=True,
                    )
                    ax.plot(t, Pred[j], "k--")
                    if true_param_prefix is not None:
                        ax.plot(t, true_p[j], "k--")
                    ax.set_xlabel("time (" + unit + ")")
                    if y_log_scale:
                        ax.set_yscale("log")
                    if log_unnormalized:
                        ax.set_ylabel("Expression (log)")
                    else:
                        ax.set_ylabel("Expression")
                    ax.set_title(gene_name + " " + title_[j])
            elif experiment_type is "one_shot":
                if has_splicing:
                    layers = (
                        ["X_uu", "X_ul", "X_su", "X_sl"]
                        if "X_ul" in adata.layers.keys()
                        else ["uu", "ul", "su", "sl"]
                    )
                    uu, ul, su, sl = (
                        adata[:, gene_name].layers[layers[0]],
                        adata[:, gene_name].layers[layers[1]],
                        adata[:, gene_name].layers[layers[2]],
                        adata[:, gene_name].layers[layers[3]],
                    )
                    uu, ul, su, sl = (
                        (
                            uu.toarray().squeeze(),
                            ul.toarray().squeeze(),
                            su.toarray().squeeze(),
                            sl.toarray().squeeze(),
                        )
                        if issparse(uu)
                        else (uu.squeeze(), ul.squeeze(), su.squeeze(), sl.squeeze())
                    )

                    if log_unnormalized and layers == ["uu", "ul", "su", "sl"]:
                        uu, ul, su, sl = (
                            np.log(uu + 1),
                            np.log(ul + 1),
                            np.log(su + 1),
                            np.log(sl + 1),
                        )

                    alpha, beta, gamma, U0, S0 = adata.var.loc[
                        gene_name,
                        [
                            prefix + "alpha",
                            prefix + "beta",
                            prefix + "gamma",
                            prefix + "U0",
                            prefix + "S0",
                        ],
                    ]
                    # $u$ - unlabeled, unspliced
                    # $s$ - unlabeled, spliced
                    # $w$ - labeled, unspliced
                    # $l$ - labeled, spliced
                    #
                    beta = sys.float_info.epsilon if beta == 0 else beta
                    gamma = sys.float_info.epsilon if gamma == 0 else gamma
                    U_sol = sol_u(t, U0, 0, beta)
                    S_sol = sol_u(t, S0, 0, gamma)
                    l = sol_u(t, 0, alpha, beta) + sol_s(t, 0, 0, alpha, beta, gamma)
                    L = sl + ul
                    if true_param_prefix is not None:
                        true_alpha, true_beta, true_gamma = (
                            adata.var.loc[gene_name, true_param_prefix + "alpha"]
                            if true_param_prefix + "alpha" in adata.var_keys()
                            else -np.inf,
                            adata.var.loc[gene_name, true_param_prefix + "beta"]
                            if true_param_prefix + "beta" in adata.var_keys()
                            else -np.inf,
                            adata.var.loc[gene_name, true_param_prefix + "gamma"]
                            if true_param_prefix + "gamma" in adata.var_keys()
                            else -np.inf,
                        )
                        true_l = sol_u(t, 0, true_alpha, true_beta) + sol_s(
                            t, 0, 0, true_alpha, true_beta, true_gamma
                        )

                    title_ = ["labeled"]
                else:
                    layers = (
                        ["X_new", "X_total"]
                        if "X_new" in adata.layers.keys()
                        else ["new", "total"]
                    )
                    uu, ul = (
                        adata[:, gene_name].layers[layers[1]]
                        - adata[:, gene_name].layers[layers[0]],
                        adata[:, gene_name].layers[layers[0]],
                    )
                    uu, ul = (
                        (uu.toarray().squeeze(), ul.toarray().squeeze())
                        if issparse(uu)
                        else (uu.squeeze(), ul.squeeze())
                    )

                    if log_unnormalized and layers == ["new", "total"]:
                        uu, ul = np.log(uu + 1), np.log(ul + 1)

                    alpha, gamma, total0 = adata.var.loc[
                        gene_name,
                        [prefix + "alpha", prefix + "gamma", prefix + "total0"],
                    ]

                    # require no beta functions
                    old = sol_u(t, total0, 0, gamma)
                    s = None  # sol_s(t, su0, uu0, 0, 1, gamma)
                    w = None
                    l = sol_u(t, 0, alpha, gamma)  # sol_s(t, 0, 0, alpha, 1, gamma)
                    L = ul
                    if true_param_prefix is not None:
                        true_alpha, true_gamma = (
                            adata.var.loc[gene_name, true_param_prefix + "alpha"]
                            if true_param_prefix + "alpha" in adata.var_keys()
                            else -np.inf,
                            adata.var.loc[gene_name, true_param_prefix + "gamma"]
                            if true_param_prefix + "gamma" in adata.var_keys()
                            else -np.inf,
                        )
                        true_l = sol_u(
                            t, 0, true_alpha, true_gamma
                        )  # sol_s(t, 0, 0, alpha, 1, gamma)

                    title_ = ["labeled"]

                Obs, Pred = np.hstack((np.zeros(L.shape), L)), np.hstack(l)
                if true_param_prefix is not None:
                    true_p = np.hstack(true_l)

                row_ind = int(
                    np.floor(i / ncols)
                )  # make sure unlabled and labeled are in the same column.
                ax = plt.subplot(
                    gs[
                        (row_ind * sub_plot_n) * ncols * grp_len
                        + (i % ncols - 1) * grp_len
                        + 1
                    ]
                )
                if true_param_prefix is not None:
                    if has_splicing:
                        ax.text(
                            0.05,
                            0.90,
                            r"$\alpha$ "
                            + ": {0:.2f}; ".format(true_alpha)
                            + r"$\hat \alpha$"
                            + ": {0:.2f} \n".format(alpha)
                            + r"$\beta$"
                            + ": {0:.2f}; ".format(true_beta)
                            + r"$\hat \beta$"
                            + ": {0:.2f} \n".format(beta)
                            + r"$\gamma$"
                            + ": {0:.2f}; ".format(true_gamma)
                            + r"$\hat \gamma$"
                            + ": {0:.2f} \n".format(gamma),
                            ha="left",
                            va="top",
                            transform=ax.transAxes,
                        )
                    else:
                        ax.text(
                            0.05,
                            0.90,
                            r"$\alpha$"
                            + ": {0:.2f}; ".format(true_alpha)
                            + r"$\hat \alpha$"
                            + ": {0:.2f} \n".format(alpha)
                            + r"$\gamma$"
                            + ": {0:.2f}; ".format(true_gamma)
                            + r"$\hat \gamma$"
                            + ": {0:.2f} \n".format(gamma),
                            ha="left",
                            va="top",
                            transform=ax.transAxes,
                        )
                ax.boxplot(
                    x=[
                        Obs[np.hstack((np.zeros_like(T), T)) == std]
                        for std in [0, T_uniq[0]]
                    ],
                    positions=[0, T_uniq[0]],
                    widths=boxwidth,
                    showfliers=False,
                    showmeans=True,
                )
                ax.plot(t, Pred, "k--")
                if true_param_prefix is not None:
                    ax.plot(t, true_p, "r--")
                ax.set_xlabel("time (" + unit + ")")
                if y_log_scale:
                    ax.set_yscale("log")
                if log_unnormalized:
                    ax.set_ylabel("Expression (log)")
                else:
                    ax.set_ylabel("Expression")
                ax.set_title(gene_name + " " + title_[0])
            elif experiment_type is "mix_std_stm":
                if has_splicing:
                    layers = (
                        ["X_uu", "X_ul", "X_su", "X_sl"]
                        if "X_ul" in adata.layers.keys()
                        else ["uu", "ul", "su", "sl"]
                    )
                    uu, ul, su, sl = (
                        adata[:, gene_name].layers[layers[0]],
                        adata[:, gene_name].layers[layers[1]],
                        adata[:, gene_name].layers[layers[2]],
                        adata[:, gene_name].layers[layers[3]],
                    )
                    uu, ul, su, sl = (
                        (
                            uu.toarray().squeeze(),
                            ul.toarray().squeeze(),
                            su.toarray().squeeze(),
                            sl.toarray().squeeze(),
                        )
                        if issparse(uu)
                        else (uu.squeeze(), ul.squeeze(), su.squeeze(), sl.squeeze())
                    )

                    if log_unnormalized and layers == ["uu", "ul", "su", "sl"]:
                        uu, ul, su, sl = (
                            np.log(uu + 1),
                            np.log(ul + 1),
                            np.log(su + 1),
                            np.log(sl + 1),
                        )

                    beta, gamma, alpha_std = adata.var.loc[
                        gene_name,
                        [prefix + "beta", prefix + "gamma", prefix + "alpha_std"],
                    ]
                    alpha_stm = adata[:, gene_name].varm[prefix + "alpha"].flatten()[1:]
                    alpha_stm0, k, _ = solve_first_order_deg(T_uniq[1:], alpha_stm)

                    # $u$ - unlabeled, unspliced
                    # $s$ - unlabeled, spliced
                    # $w$ - labeled, unspliced
                    # $l$ - labeled, spliced
                    #
                    # calculate labeled unspliced and spliced mRNA amount
                    u1, s1, u1_, s1_ = (
                        np.zeros(len(t) - 1),
                        np.zeros(len(t) - 1),
                        np.zeros(len(T_uniq) - 1),
                        np.zeros(len(T_uniq) - 1),
                    )
                    for ind in np.arange(1, len(t)):
                        t_i = t[ind]
                        u0 = sol_u(np.max(t) - t_i, 0, alpha_std, beta)
                        alpha_stm_t_i = alpha_stm0 * np.exp(-k * t_i)
                        u1[ind - 1], s1[ind - 1] = (
                            sol_u(t_i, u0, alpha_stm_t_i, beta),
                            sol_u(np.max(t), 0, beta, gamma),
                        )
                    for ind in np.arange(1, len(T_uniq)):
                        t_i = T_uniq[ind]
                        u0 = sol_u(np.max(T_uniq) - t_i, 0, alpha_std, beta)
                        alpha_stm_t_i = alpha_stm[ind - 1]
                        u1_[ind - 1], s1_[ind - 1] = (
                            sol_u(t_i, u0, alpha_stm_t_i, beta),
                            sol_u(np.max(T_uniq), 0, beta, gamma),
                        )

                    Obs, Pred, Pred_ = (
                        np.vstack((ul, sl, uu, su)),
                        np.vstack((u1.reshape(1, -1), s1.reshape(1, -1))),
                        np.vstack((u1_.reshape(1, -1), s1_.reshape(1, -1))),
                    )
                    j_species, title_ = (
                        4,
                        [
                            "unspliced labeled (new)",
                            "spliced labeled (new)",
                            "unspliced unlabeled (old)",
                            "spliced unlabeled (old)",
                            "alpha (steady state vs. stimulation)",
                        ],
                    )
                else:
                    layers = (
                        ["X_new", "X_total"]
                        if "X_new" in adata.layers.keys()
                        else ["new", "total"]
                    )
                    uu, ul = (
                        adata[:, gene_name].layers[layers[1]]
                        - adata[:, gene_name].layers[layers[0]],
                        adata[:, gene_name].layers[layers[0]],
                    )
                    uu, ul = (
                        (uu.toarray().squeeze(), ul.toarray().squeeze())
                        if issparse(uu)
                        else (uu.squeeze(), ul.squeeze())
                    )

                    if log_unnormalized and layers == ["new", "total"]:
                        uu, ul = np.log(uu + 1), np.log(ul + 1)

                    gamma, alpha_std = adata.var.loc[
                        gene_name, [prefix + "gamma", prefix + "alpha_std"]
                    ]
                    alpha_stm = adata[:, gene_name].varm[prefix + "alpha"].flatten()[1:]

                    alpha_stm0, k, _ = solve_first_order_deg(T_uniq[1:], alpha_stm)
                    # require no beta functions
                    u1, u1_ = (
                        np.zeros(len(t) - 1),
                        np.zeros(len(T_uniq) - 1),
                    )  # interpolation or original time point
                    for ind in np.arange(1, len(t)):
                        t_i = t[ind]
                        u0 = sol_u(np.max(t) - t_i, 0, alpha_std, gamma)
                        alpha_stm_t_i = alpha_stm0 * np.exp(-k * t_i)
                        u1[ind - 1] = sol_u(t_i, u0, alpha_stm_t_i, gamma)

                    for ind in np.arange(1, len(T_uniq)):
                        t_i = T_uniq[ind]
                        u0 = sol_u(np.max(T_uniq) - t_i, 0, alpha_std, gamma)
                        alpha_stm_t_i = alpha_stm[ind - 1]
                        u1_[ind - 1] = sol_u(t_i, u0, alpha_stm_t_i, gamma)

                    Obs, Pred, Pred_ = (
                        np.vstack((ul, uu)),
                        np.vstack((u1.reshape(1, -1))),
                        np.vstack((u1_.reshape(1, -1))),
                    )
                    j_species, title_ = (
                        2,
                        [
                            "labeled (new)",
                            "unlabeled (old)",
                            "alpha (steady state vs. stimulation)",
                        ],
                    )

                group_list = [np.repeat(alpha_std, len(alpha_stm)), alpha_stm]

                for j in range(sub_plot_n):
                    row_ind = int(
                        np.floor(i / ncols)
                    )  # make sure all related plots for the same gene in the same column.
                    ax = plt.subplot(
                        gs[
                            (row_ind * sub_plot_n + j) * ncols * grp_len
                            + (i % ncols - 1) * grp_len
                            + 1
                        ]
                    )
                    if j < j_species / 2:
                        ax.boxplot(
                            x=[Obs[j][T == std] for std in T_uniq[1:]],
                            positions=T_uniq[1:],
                            widths=boxwidth,
                            showfliers=False,
                            showmeans=True,
                        )
                        ax.plot(t[1:], Pred[j], "k--")
                        ax.scatter(T_uniq[1:], Pred_[j], s=20, c="red")
                        ax.set_xlabel("time (" + unit + ")")
                        ax.set_title(gene_name + ": " + title_[j])

                        if y_log_scale:
                            ax.set_yscale("log")
                        if log_unnormalized:
                            ax.set_ylabel("Expression (log)")
                        else:
                            ax.set_ylabel("Expression")
                    elif j < j_species:
                        ax.boxplot(
                            x=[Obs[j][T == std] for std in T_uniq],
                            positions=T_uniq,
                            widths=boxwidth,
                            showfliers=False,
                            showmeans=True,
                        )
                        ax.set_xlabel("time (" + unit + ")")
                        ax.set_title(gene_name + ": " + title_[j])

                        if y_log_scale:
                            ax.set_yscale("log")
                        if log_unnormalized:
                            ax.set_ylabel("Expression (log)")
                        else:
                            ax.set_ylabel("Expression")

                    else:
                        x = T_uniq[1:]  # the label locations
                        group_width = barwidth / 2
                        bar_coord, group_name, group_ind = (
                            [-1, 1],
                            ["steady state", "stimulation"],
                            0,
                        )

                        for group_ind in range(len(group_list)):
                            cur_group = group_list[group_ind]
                            ax.bar(
                                x + bar_coord[group_ind] * group_width,
                                cur_group,
                                barwidth,
                                label=group_name[group_ind],
                            )
                            # Add gene name, experimental type, etc.
                            ax.set_xlabel("time (" + unit + ")")
                            ax.set_ylabel("alpha (translation rate)")
                            ax.set_xticks(x)
                            ax.set_xticklabels(x)
                            group_ind += 1
                        ax.legend()

                    ax.set_xlabel("time (" + unit + ")")
                    ax.set_title(gene_name + ": " + title_[j])
            elif experiment_type is "multi_time_series":
                pass  # group by different groups
            elif experiment_type is "coassay":
                pass  # show protein velocity (steady state and the Gamma distribution model)

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'dynamics', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return g


def dynamics_(
    adata,
    gene_names,
    color,
    dims=[0, 1],
    current_layer="spliced",
    use_raw=False,
    Vkey="S",
    Ekey="spliced",
    basis="umap",
    mode="all",
    cmap=None,
    gs=None,
    **kwargs
):
    """

    Parameters
    ----------
    adata
    basis
    mode: `str` (default: all)
        Support mode includes: phase, expression, velocity, all

    Returns
    -------

    """

    import matplotlib.pyplot as plt

    genes = list(set(gene_names).intersection(adata.var.index))
    for i, gn in enumerate(genes):
        ax = plt.subplot(gs[i * 3])
        try:
            ix = np.where(adata.var["Gene"] == gn)[0][0]
        except:
            continue

        scatters(
            adata,
            gene_names,
            color,
            dims=[0, 1],
            current_layer="spliced",
            use_raw=False,
            Vkey="S",
            Ekey="spliced",
            basis="umap",
            mode="all",
            cmap=None,
            gs=None,
            **kwargs
        )

        scatters(
            adata,
            gene_names,
            color,
            dims=[0, 1],
            current_layer="spliced",
            use_raw=False,
            Vkey="S",
            Ekey="spliced",
            basis="umap",
            mode="all",
            cmap=None,
            gs=None,
            **kwargs
        )

        scatters(
            adata,
            gene_names,
            color,
            dims=[0, 1],
            current_layer="spliced",
            use_raw=False,
            Vkey="S",
            Ekey="spliced",
            basis="umap",
            mode="all",
            cmap=None,
            gs=None,
            **kwargs
        )

    plt.tight_layout()
