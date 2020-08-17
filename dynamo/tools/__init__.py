"""Mapping Vector Field of Single Cells
"""

# inclusive expression dynamics model related
# from .dynamo import sol_u, sol_s, fit_gamma_labelling, fit_alpha_labelling, fit_gamma_splicing, fit_gamma
# from .dynamo_fitting import sol_u, sol_s, sol_p, sol_ode, sol_num, fit_gamma_labelling, fit_beta_lsq, fit_alpha_labelling, fit_alpha_synthesis, fit_gamma_splicing, fit_gamma


from .moments import moments, calc_1nd_moment, calc_2nd_moment

from .cell_vectors import (
    cell_velocities,
    confident_cell_velocities,
    generalized_diffusion_map,
    stationary_distribution,
    diffusion,
    expected_return_time,
    embed_velocity,
)

from .dynamics import dynamics

# run other velocity tools:
from .velocyto_scvelo import (
    vlm_to_adata,
    converter,
    run_velocyto,
    run_scvelo,
    mean_var_by_time,
    run_dynamo,
    run_dynamo_simple_fit,
    run_dynamo_labelling,
    compare_res,
)

# vector field related
from .metric_velocity import (
    cell_wise_confidence,
    gene_wise_confidence,
)

# Markov chain related:
from .Markov import (
    markov_combination,
    compute_markov_trans_prob,
    compute_kernel_trans_prob,
    compute_drift_kernel,
    compute_drift_local_kernel,
    compute_density_kernel,
    makeTransitionMatrix,
    compute_tau,
    MarkovChain,
    KernelMarkovChain,
    DiscreteTimeMarkovChain,
    ContinuousTimeMarkovChain,
)

# state graph related

# dimension reduction related
from .dimension_reduction import reduceDimension  # , run_umap

# clustering related
from .clustering import hdbscan, cluster_field

# mnn related
from .connectivity import mnn, neighbors

# Pseudotime related
from .DDRTree_py import DDRTree
from .DDRTree_py import cal_ncenter
from .psl_py import psl

# DEG test related
from .markers import (
    moran_i,
    find_group_markers,
    two_groups_degs,
    top_n_markers,
    glm_degs,
)

# Sampling methods
from .sampling import (
    TRNET,
    trn,
    sample_by_velocity,
    lhsclassic,
    sample
)

