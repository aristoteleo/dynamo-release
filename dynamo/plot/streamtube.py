from numbers import Number

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype

from ..configuration import _themes
from ..tools.Markov import prepare_velocity_grid_data
from .utils import _to_hex, is_cell_anno_column, is_gene_name


def plot_3d_streamtube(
    adata,
    color,
    layer,
    group,
    init_group,
    basis="umap",
    dims=[0, 1, 2],
    theme=None,
    background=None,
    cmap=None,
    color_key=None,
    color_key_cmap=None,
    html_fname=None,
    save_show_or_return="show",
    save_kwargs={},
):
    """Plot a interative 3d streamtube plot via plotly.

    A streamtube is a tubular region surrounded by streamlines that form a closed loop. It's a continuous version of a
    streamtube plot (3D quiver plot) and can provide insight into flow data from natural systems. The color of tubes is
    determined by their local norm, and the diameter of the field by the local divergence of the vector field.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            An Annodata object, must have vector field reconstructed for the input `basis` whose dimension should at
            least 3D.
        color: `string` (default: `ntr`)
            Any column names or gene expression, etc. that will be used for coloring cells.
        group: `str`
            The column names of adata.obs that will be used to search for cells, together with `init_group` to set the
            initial state of the streamtube.
        init_group: `str`
            The group name among all names in `group` that will be used to set the initial states of the stream tube.
        basis: `str`
            The reduced dimension.
        html_fname: `str` or None
            html file name that will be use to save the streamtube interactive plot.
        dims: `list` (default: `[0, 1, 2]`)
            The number of dimensions that will be used to construct the vector field for streamtube plot.
        save_show_or_return: `str` {'save', 'show', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.

    Returns
    -------
        Nothing but render an interactive streamtube plot. If html_fname is not None, the plot will save to a html file.
    """

    try:
        # 3D streamtube:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("You need to install the package `plotly`. Install hiveplotlib via `pip install plotly`")

    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    if is_gene_name(adata, color):
        color_val = adata.obs_vector(k=color, layer=None) if layer == "X" else adata.obs_vector(k=color, layer=layer)
    elif is_cell_anno_column(adata, color):
        color_val = adata.obs_vector

    is_not_continous = not isinstance(color_val[0], Number) or color_val.dtype.name == "category"

    if is_not_continous:
        labels = color_val.to_dense() if is_categorical_dtype(color_val) else color_val
        if theme is None:
            if _background in ["#ffffff", "black"]:
                _theme_ = "glasbey_dark"
            else:
                _theme_ = "glasbey_white"
        else:
            _theme_ = theme
    else:
        values = color_val
        if theme is None:
            if _background in ["#ffffff", "black"]:
                _theme_ = "inferno" if not layer.startswith("velocity") else "div_blue_black_red"
            else:
                _theme_ = "viridis" if not layer.startswith("velocity") else "div_blue_red"
        else:
            _theme_ = theme

    _cmap = _themes[_theme_]["cmap"] if cmap is None else cmap
    _color_key_cmap = _themes[_theme_]["color_key_cmap"] if color_key_cmap is None else color_key_cmap

    if is_not_continous:
        labels = adata.obs[color]
        unique_labels = labels.unique()
        if isinstance(color_key, dict):
            colors = pd.Series(labels).map(color_key).values
        else:
            color_key = _to_hex(plt.get_cmap(color_key_cmap)(np.linspace(0, 1, len(unique_labels))))

            new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
            colors = pd.Series(labels).map(new_color_key)
    else:
        norm = matplotlib.colors.Normalize(vmin=np.min(values), vmax=np.max(values), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=_cmap)
        colors = _to_hex(mapper.to_rgba(values))

    X = adata.obsm["X_" + basis][:, dims]
    grid_kwargs_dict = {
        "density": None,
        "smooth": None,
        "n_neighbors": None,
        "min_mass": None,
        "autoscale": False,
        "adjust_for_stream": True,
        "V_threshold": None,
    }

    X_grid, p_mass, neighs, weight = prepare_velocity_grid_data(
        X,
        [60, 60, 60],
        density=grid_kwargs_dict["density"],
        smooth=grid_kwargs_dict["smooth"],
        n_neighbors=grid_kwargs_dict["n_neighbors"],
    )

    from .vectorfield.utils import vecfld_from_adata

    VecFld, func = vecfld_from_adata(adata, basis="umap")

    velocity_grid = func(X_grid)

    fig = go.Figure(
        data=go.Streamtube(
            x=X_grid[:, 0],
            y=X_grid[:, 1],
            z=X_grid[:, 2],
            u=velocity_grid[:, 0],
            v=velocity_grid[:, 1],
            w=velocity_grid[:, 2],
            starts=dict(
                x=adata[labels == init_group, :].obsm["X_umap"][:125, 0],
                y=adata[labels == init_group, :].obsm["X_umap"][:125, 1],
                z=adata[labels == init_group, :].obsm["X_umap"][:125, 2],
            ),
            sizeref=3000,
            colorscale="Portland",
            showscale=False,
            maxdisplayed=3000,
        )
    )

    fig.update_layout(
        scene=dict(
            aspectratio=dict(
                x=2,
                y=1,
                z=1,
            )
        ),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    fig.add_scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode="markers",
        marker=dict(size=2, color=colors.values),
    )

    if save_show_or_return == "save" or html_fname is not None:
        html_fname = "streamtube_" + color + "_" + group + "_" + init_group if html_fname is None else html_fname
        save_kwargs_ = {"file": html_fname, "auto_open": True}
        save_kwargs_.update(save_kwargs)
        fig.write_html(**save_kwargs_)
    elif save_show_or_return == "show":
        fig.show()
    elif save_show_or_return == "return":
        return fig
