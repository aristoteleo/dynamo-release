"""Mapping Vector Field of Single Cells
"""

from .Ao import Ao_pot_map, solveQ
from .Bhattacharya import alignment, path_integral
from .cell_vectors import cell_accelerations, cell_curvatures

# vector field clustering related:
from .clustering import cluster_field, streamline_clusters
from .networks import adj_list_to_matrix, build_network_per_cluster
from .rank_vf import (
    aggregateRegEffs,
    rank_acceleration_genes,
    rank_cell_groups,
    rank_curvature_genes,
    rank_divergence_genes,
    rank_expression_genes,
    rank_genes,
    rank_jacobian_genes,
    rank_s_divergence_genes,
    rank_sensitivity_genes,
    rank_velocity_genes,
)

# potential related
from .scPotential import (  # , vector_field_function
    DiffusionMatrix,
    IntGrad,
    Pot,
    Potential,
    action,
    gen_fixed_points,
    gen_gradient,
    search_fixed_points,
)
from .scVectorField import (  # , evaluate, con_K_div_cur_free
    BaseVectorField,
    BifurcationTwoGenesVectorField,
    SparseVFC,
    SvcVectorField,
    graphize_vecfld,
)

# stochastic process related
from .stochastic_process import diffusionMatrix
from .topography import (
    FixedPoints,
    VectorField,
    VectorField2D,
    assign_fixedpoints,
    topography,
)
from .utils import get_jacobian, parse_int_df, vector_field_function
from .vector_calculus import (
    acceleration,
    curl,
    curvature,
    divergence,
    hessian,
    jacobian,
    laplacian,
    sensitivity,
    speed,
    torsion,
    velocities,
)

# vfGraph operation related:
from .vfGraph_deprecated import vfGraph
from .Wang import MFPT, Wang_action, Wang_LAP, transition_rate
