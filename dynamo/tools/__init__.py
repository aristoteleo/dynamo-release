"""Mapping Vector Field of Single Cells
"""

# inclusive expression dynamics model related
# from .dynamo import sol_u, sol_s, fit_gamma_labelling, fit_alpha_labelling, fit_gamma_splicing, fit_gamma
# from .dynamo_fitting import sol_u, sol_s, sol_p, sol_ode, sol_num, fit_gamma_labelling, fit_beta_lsq, fit_alpha_labelling, fit_alpha_synthesis, fit_gamma_splicing, fit_gamma


from .cell_velocities import (
    cell_velocities,
    confident_cell_velocities,
    diffusion,
    expected_return_time,
    generalized_diffusion_map,
    stationary_distribution,
)

# clustering related
from .clustering import (
    cluster_community,
    cluster_community_from_graph,
    hdbscan,
    leiden,
    louvain,
    purity,
    scc,
)

# mnn related
from .connectivity import (
    check_and_recompute_neighbors,
    check_neighbors_completeness,
    mnn,
    neighbors,
)

# Pseudotime related
from .DDRTree_graph import construct_velocity_tree, directed_pg
from .DDRTree import DDRTree, cal_ncenter
from .pseudotime import order_cells

# dimension reduction related
from .dimension_reduction import reduceDimension  # , run_umap
from .dynamics import dynamics

# state graph related
from .graph_calculus import GraphVectorField

# cell proliferation and death:
from .growth import cell_growth_rate, growth_rate, n_descentants, score_cells

# DEG test related
from .markers import (
    find_group_markers,
    glm_degs,
    moran_i,
    top_n_markers,
    two_groups_degs,
)

# Markov chain related:
from .Markov import (
    ContinuousTimeMarkovChain,
    DiscreteTimeMarkovChain,
    KernelMarkovChain,
    MarkovChain,
    compute_density_kernel,
    compute_drift_kernel,
    compute_drift_local_kernel,
    compute_kernel_trans_prob,
    compute_markov_trans_prob,
    compute_tau,
    makeTransitionMatrix,
    markov_combination,
)

# vector field related
from .metric_velocity import cell_wise_confidence, gene_wise_confidence
from .moments import calc_1nd_moment, calc_2nd_moment, moments
from .pseudotime_velocity import pseudotime_velocity
from .psl import psl

# recipes:
from .recipes import (
    recipe_deg_data,
    recipe_kin_data,
    recipe_mix_kin_deg_data,
    recipe_one_shot_data,
    velocity_N,
)

# Sampling methods
from .sampling import TRNET, lhsclassic, sample, sample_by_kmeans, sample_by_velocity, trn
from .utils import (
    AnnDataPredicate,
    cell_norm,
    compute_smallest_distance,
    get_vel_params,
    index_gene,
    select,
    select_cell,
    table_rank_dict,
)

# run other velocity tools:
from .velocyto_scvelo import (
    converter,
    mean_var_by_time,
    run_scvelo,
    run_velocyto,
    scv_dyn_convertor,
    vlm_to_adata,
)

# deprecated functions
from .deprecated import (
    construct_velocity_tree_py,
)
