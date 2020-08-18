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
    exp_by_groups,
)
from .cell_cycle import cell_cycle_scores

from .dynamics import phase_portraits, dynamics
from .time_series import kinetic_curves, kinetic_heatmap, jacobian_kinetics

from .dimension_reduction import pca, tsne, umap, trimap
from .connectivity import nneighbors

from .scVectorField import (
    cell_wise_vectors,
    grid_vectors,
    streamline_plot,
    line_integral_conv,
    plot_energy
)  # , plot_LIC_gray
from .topography import (
    plot_flow_field,
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
)
from .fate import (
    fate_bias,
)
from .state_graph import state_graph

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
