from numbers import Number
from typing import Any, Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from pandas.api.types import is_categorical_dtype

from ..configuration import _themes
from ..tools.Markov import prepare_velocity_grid_data
from .utils import _to_hex, is_cell_anno_column, is_gene_name


def plot_3d_streamtube(
    adata: AnnData,
    color: str,
    layer: str,
    group: str,
    init_group: str,
    basis: str = "umap",
    dims: List[int] = [0, 1, 2],
    theme: Optional[str] = None,
    background: Optional[str] = None,
    cmap: Optional[str] = None,
    color_key: Union[Dict[str, str], List[str], None] = None,
    color_key_cmap: Optional[str] = None,
    html_fname: Optional[str] = None,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
):
    """Plot an interative 3d streamtube plot via plotly.

    A streamtube is a tubular region surrounded by streamlines that form a closed loop. It's a continuous version of a
    streamtube plot (3D quiver plot) and can provide insight into flow data from natural systems. The color of tubes is
    determined by their local norm, and the diameter of the field by the local divergence of the vector field.

    Args:
        adata: an Annodata object, must have vector field reconstructed for the input `basis` whose dimension should at
            least 3D.
        color: any column names or gene expression, etc. that will be used for coloring cells.
        layer: the layer key of the expression data.
        group: the column names of adata.obs that will be used to search for cells, together with `init_group` to set
            the initial state of the streamtube.
        init_group: the group name among all names in `group` that will be used to set the initial states of the stream
            tube.
        basis: the reduced dimension. Defaults to "umap".
        dims: the number of dimensions that will be used to construct the vector field for streamtube plot. Defaults to
            [0, 1, 2].
        theme: the theme of the plot. Defaults to None.
        background: the background color of the plot. Defaults to None.
        cmap: The name of a matplotlib colormap to use for coloring the plots. Defaults to None.
        color_key: the method to assign colors to categoricals. Defaults to None.
        color_key_cmap: the name of a matplotlib colormap to use for categorical coloring. Defaults to None.
        html_fname: html file name that will be used to save the streamtube interactive plot. Defaults to None.
        save_show_or_return: whether to save, show, or return the figures. Defaults to "show".
        save_kwargs:  A dictionary that will be passed to the save_fig function. By default, it is an empty dictionary
            and the save_fig function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.

    Raises:
        ImportError: plotly is not installed.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the generated plotly figure
        would be returned.
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

    color_val = adata.obs_vector(k=color, layer=None) if layer == "X" else adata.obs_vector(k=color, layer=layer)

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

    if adata.obsm["X_" + basis].shape[1] < 3:
        raise ValueError("Current basis has dimensions less than 3!")

    if "VecFld_" + basis not in adata.uns.keys():
        raise KeyError("Corresponding vector field not found! Please run VectorField() with current basis.")

    X = adata.obsm["X_" + basis][:, dims]

    if "grid" in adata.uns["VecFld_" + basis].keys() and "grid_V" in adata.uns["VecFld_" + basis].keys():
        X_grid = adata.uns["VecFld_" + basis]["grid"]
        velocity_grid = adata.uns["VecFld_" + basis]["grid_V"]
    else:
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

        from ..vectorfield.utils import vecfld_from_adata

        VecFld, func = vecfld_from_adata(adata, basis=basis)

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
                x=adata[labels == init_group, :].obsm["X_" + basis][:125, 0],
                y=adata[labels == init_group, :].obsm["X_" + basis][:125, 1],
                z=adata[labels == init_group, :].obsm["X_" + basis][:125, 2],
            ),
            colorscale="Portland",
            showscale=False,
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

    if (save_show_or_return in ["save", "both", "all"]) or html_fname is not None:
        html_fname = "streamtube_" + color + "_" + group + "_" + init_group if html_fname is None else html_fname
        save_kwargs_ = {"file": html_fname, "auto_open": True}
        save_kwargs_.update(save_kwargs)
        fig.write_html(**save_kwargs_)
    if save_show_or_return in ["show", "both", "all"]:
        fig.show()
    if save_show_or_return in ["return", "all"]:
        return fig
