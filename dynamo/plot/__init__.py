"""Mapping Vector Field of Single Cells
"""

# from .theme import points
from .utils import quiver_autoscaler, save_fig
from .scatters import scatters

from .preprocess import (
    basic_stats,
    show_fraction,
    feature_genes,
    variance_explained,
    biplot,
    loading,
    exp_by_groups,
    highest_frac_genes,
)
from .cell_cycle import cell_cycle_scores
from .markers import bubble

from .dynamics import phase_portraits, dynamics
from .time_series import (
    kinetic_curves,
    kinetic_heatmap,
    jacobian_kinetics,
    sensitivity_kinetics,
)

from .dimension_reduction import pca, tsne, umap, trimap
from .connectivity import nneighbors

from .scVectorField import (
    cell_wise_vectors,
    grid_vectors,
    streamline_plot,
    line_integral_conv,
    plot_energy,
)  # , plot_LIC_gray
from .streamtube import plot_3d_streamtube
from .topography import (
    plot_flow_field,
    plot_fixed_points_2d,
    plot_fixed_points,
    plot_nullclines,
    plot_separatrix,
    plot_traj,
    topography,
)
from .vector_calculus import (
    speed,
    curl,
    divergence,
    curvature,
    jacobian,
    jacobian_heatmap,
    sensitivity,
    sensitivity_heatmap,
)
from .networks import (
    arcPlot,
    circosPlot,
    hivePlot,
)

from .fate import fate_bias, fate
from .state_graph import state_graph
from .least_action_path import least_action, lap_min_time

from .scPotential import show_landscape

from .ezplots import (
    zscatter,
    zstreamline,
    multiplot,
    plot_V,
    plot_X,
    SchemeDiverge,
    SchemeDivergeBWR,
)

from .clustering import louvain, leiden, infomap, streamline_clusters
from .heatmaps import response, plot_hill_function, causality, comb_logic

# spatial data related
from .space import space

__all__ = [
    "quiver_autoscaler",
    "save_fig",
    "scatters",
    "basic_stats",
    "show_fraction",
    "feature_genes",
    "variance_explained",
    "biplot",
    "loading",
    "exp_by_groups",
    "highest_frac_genes",
    "cell_cycle_scores",
    "bubble",
    "phase_portraits",
    "dynamics",
    "kinetic_curves",
    "kinetic_heatmap",
    "jacobian_kinetics",
    "sensitivity_kinetics",
    "pca",
    "tsne",
    "umap",
    "trimap",
    "nneighbors",
    "cell_wise_vectors",
    "grid_vectors",
    "streamline_plot",
    "line_integral_conv",
    "plot_energy",
    "plot_3d_streamtube",
    "plot_flow_field",
    "plot_fixed_points_2d",
    "plot_fixed_points",
    "plot_nullclines",
    "plot_separatrix",
    "plot_traj",
    "topography",
    "speed",
    "curl",
    "divergence",
    "curvature",
    "jacobian",
    "jacobian_heatmap",
    "sensitivity",
    "sensitivity_heatmap",
    "arcPlot",
    "circosPlot",
    "hivePlot",
    "fate_bias",
    "fate",
    "state_graph",
    "least_action",
    "lap_min_time",
    "show_landscape",
    "louvain",
    "leiden",
    "infomap",
    "space",
    "zscatter",
    "zstreamline",
    "multiplot",
    "plot_V",
    "plot_X",
    "SchemeDiverge",
    "SchemeDivergeBWR",
    "streamline_clusters",
    "response",
    "plot_hill_function",
    "causality",
    "comb_logic",
]
