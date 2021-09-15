"""Mapping Vector Field of Single Cells
"""

from .scVectorField import (
    SparseVFC,
    graphize_vecfld,
    BaseVectorField,
    SvcVectorField,
)  # , evaluate, con_K_div_cur_free
from .utils import (
    vector_field_function,
    parse_int_df,
    get_jacobian,
)
from .topography import (
    FixedPoints,
    VectorField2D,
    topography,
    VectorField,
    assign_fixedpoints,
)

from .cell_vectors import cell_accelerations, cell_curvatures
from .vector_calculus import (
    velocities,
    speed,
    jacobian,
    sensitivity,
    curl,
    divergence,
    acceleration,
    curvature,
    torsion,
    rank_expression_genes,
    rank_velocity_genes,
    rank_divergence_genes,
    rank_s_divergence_genes,
    rank_acceleration_genes,
    rank_curvature_genes,
    rank_genes,
    rank_cells,
    rank_jacobian_genes,
    rank_sensitivity_genes,
    aggregateRegEffs,
)
from .networks import (
    build_network_per_cluster,
    adj_list_to_matrix,
)

# potential related
from .scPotential import (
    search_fixed_points,
    gen_fixed_points,
    gen_gradient,
    IntGrad,
    DiffusionMatrix,
    action,
    Potential,
    Pot,
)  # , vector_field_function
from .Bhattacharya import path_integral, alignment
from .Wang import Wang_action, Wang_LAP, transition_rate, MFPT
from .Ao import Ao_pot_map, solveQ

# stochastic process related
from .stochastic_process import diffusionMatrix

# vfGraph operation related:
from .vfGraph_deprecated import vfGraph

# vector field clustering related:
from .clustering import cluster_field, streamline_clusters
