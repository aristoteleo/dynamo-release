"""Plot groups of least-action paths from cell-type transition results."""

from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_lap(
    adata: AnnData,
    transition_graph: Mapping[str, Mapping[str, Any]],
    paths: Union[str, Sequence[str]],
    title: Optional[str] = None,
    basis: str = "umap",
    color: str = "cell_type",
    color_key: Optional[Mapping[str, str]] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (3, 3),
    ax: Optional[Axes] = None,
    **streamline_kwargs,
) -> Tuple[Figure, Axes]:
    """Overlay selected least-action paths on a velocity streamline plot.

    This is the reusable version of the ``plot_lap`` helper in the Dynamo
    protocol's LAP notebook. All data are supplied explicitly; no notebook
    globals are needed. Each trajectory is colored by its own cumulative action
    values, matching the notebook (colors are not normalized across paths).

    Args:
        adata: Annotated data with the embedding and velocity information needed
            by :func:`dynamo.pl.streamline_plot`.
        transition_graph: Results returned by
            :func:`dynamo.pd.compute_cell_type_transitions`. Each selected entry
            must contain ``LAP_<basis>`` with matching ``prediction`` and
            ``action`` lists.
        paths: A transition name, e.g. ``"HSC->Meg"``, or a nonempty sequence of
            names selecting developmental, dedifferentiation or other paths.
        title: Optional title for the plot.
        basis: Embedding shared by the background and paths. The first two
            components are shown. Defaults to ``"umap"``.
        color: A single annotation or gene used to color background cells.
        color_key: Optional cell-color mapping, passed to the streamline plot.
        cmap: Colormap for action values along each path, not background cells.
        figsize: Figure size when creating a new figure.
        ax: Optional existing axes. Its figure is reused.
        **streamline_kwargs: Additional arguments to
            :func:`dynamo.pl.streamline_plot`, such as ``density`` or
            ``show_legend``. ``frontier`` defaults to True. Display/save mode
            and embedding components are controlled here; use the returned
            figure to save or show the completed plot.

    Returns:
        The matplotlib ``(figure, axes)``. The function does not call ``show``
        or save the figure automatically.

    Raises:
        KeyError: A selected transition, LAP result or embedding is missing.
        ValueError: No paths are selected, coordinates/action arrays are
            inconsistent, or multiple background color panels are requested.

    Examples:
        >>> graph, cells = dyn.pd.compute_cell_type_transitions(adata, cell_types)
        >>> fig, ax = dyn.pd.plot_lap(
        ...     adata, graph, ["HSC->Meg", "HSC->Ery"], title="Developmental LAPs"
        ... )
        >>> fig.savefig("developmental_laps.pdf")
    """
    import matplotlib.pyplot as plt

    from ..plot.scVectorField import streamline_plot
    from ..plot.utils import map2color

    paths = [paths] if isinstance(paths, str) else list(paths)
    if not paths:
        raise ValueError("paths must contain at least one transition name.")
    if not isinstance(color, str):
        raise ValueError("color must select a single annotation or gene.")
    reserved = {"x", "y", "save_show_or_return"}.intersection(streamline_kwargs)
    if reserved:
        raise ValueError("plot_lap controls these streamline arguments: " + ", ".join(sorted(reserved)))

    embedding_key = "X_" + basis
    if embedding_key not in adata.obsm:
        raise KeyError(f"Embedding {embedding_key!r} is missing from adata.obsm.")
    coords = np.asarray(adata.obsm[embedding_key])
    if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] < 2:
        raise ValueError(f"{embedding_key} must contain cells with at least two coordinates.")

    # Validate all requested paths before creating or modifying a figure. Avoid
    # zip silently dropping trajectories or action arrays in incomplete results.
    trajectories = []
    lap_key = "LAP_" + basis
    for path in paths:
        if path not in transition_graph:
            raise KeyError(f"Transition {path!r} is missing from transition_graph.")
        if lap_key not in transition_graph[path]:
            raise KeyError(f"Transition {path!r} has no {lap_key!r} result.")
        lap = transition_graph[path][lap_key]
        predictions, actions = lap["prediction"], lap["action"]
        if len(predictions) == 0 or len(predictions) != len(actions):
            raise ValueError(f"Transition {path!r} must have matching, nonempty prediction and action lists.")
        for prediction, action in zip(predictions, actions):
            prediction, action = np.asarray(prediction), np.asarray(action)
            if prediction.ndim != 2 or prediction.shape[0] == 0 or prediction.shape[1] < 2:
                raise ValueError(f"Transition {path!r} must have nonempty paths with at least two coordinates.")
            if action.ndim != 1 or len(action) != len(prediction):
                raise ValueError(f"Transition {path!r} must have one action value per path point.")
            trajectories.append((prediction, action))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    streamline_kwargs.setdefault("frontier", True)
    # The helper draws its own short coordinate axes. Suppress the background's
    # arrow annotations so newer scatters implementations do not duplicate them.
    scatter_kwargs = dict(streamline_kwargs.pop("s_kwargs_dict", {}))
    scatter_kwargs.setdefault("arrow", False)
    streamline_kwargs.setdefault("show_arrowed_spines", False)
    streamline_kwargs["s_kwargs_dict"] = scatter_kwargs
    streamline_plot(
        adata,
        basis=basis,
        color=color,
        color_key=color_key,
        ax=ax,
        save_show_or_return="return",
        **streamline_kwargs,
    )
    for prediction, action in trajectories:
        ax.scatter(*prediction[:, :2].T, c=map2color(action, cmap=cmap), zorder=5)
        ax.plot(*prediction[:, :2].T, c="k", zorder=4)

    # Preserve the notebook's short coordinate axes and labels.
    xmin, ymin = coords[:, :2].min(axis=0)
    xmax, ymax = coords[:, :2].max(axis=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side, start, end in (("bottom", xmin, xmax), ("left", ymin, ymax)):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_bounds(start, start + (end - start) / 6)
    ax.spines["left"].set_position(("data", xmin))
    ax.spines["bottom"].set_position(("data", ymin))
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"X_{basis}1", loc="left", fontsize=10)
    ax.set_ylabel(f"X_{basis}2", loc="bottom", fontsize=10)
    if title is not None:
        ax.set_title(title, fontsize=12)
    return ax.figure, ax
