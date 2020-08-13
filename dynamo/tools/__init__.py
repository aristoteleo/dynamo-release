"""Mapping Vector Field of Single Cells
"""

# inclusive expression dynamics model related
# from .dynamo import sol_u, sol_s, fit_gamma_labelling, fit_alpha_labelling, fit_gamma_splicing, fit_gamma
# from .dynamo_fitting import sol_u, sol_s, sol_p, sol_ode, sol_num, fit_gamma_labelling, fit_beta_lsq, fit_alpha_labelling, fit_alpha_synthesis, fit_gamma_splicing, fit_gamma

from .estimation_kinetic import (
    kinetic_estimation,
    Estimation_MomentDeg,
    Estimation_MomentDegNosp,
    Estimation_MomentKin,
    Estimation_MomentKinNosp,
    Estimation_DeterministicDeg,
    Estimation_DeterministicDegNosp,
    Estimation_DeterministicKinNosp,
    Estimation_DeterministicKin,
    GoodnessOfFit,
)

from .utils_kinetic import (
    LinearODE,
    Moments,
    Moments_Nosplicing,
    Moments_NoSwitching,
    Moments_NoSwitchingNoSplicing,
    Deterministic,
    Deterministic_NoSplicing,
)

from .moments import Estimation, moments, calc_1nd_moment, calc_2nd_moment

from .velocity import (
    sol_u,
    sol_s,
    sol_p,
    fit_linreg,
    fit_first_order_deg_lsq,
    solve_first_order_deg,
    fit_gamma_lsq,
    fit_alpha_synthesis,
    fit_alpha_degradation,
    velocity,
    ss_estimation,
)
from .cell_vectors import (
    cell_velocities,
    cell_accelerations,
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
from .scVectorField import (
    SparseVFC,
    graphize_vecfld,
    vectorfield
)  # , evaluate, con_K_div_cur_free
from .utils_vecCalc import (
    vector_field_function, 
)
from .topography import FixedPoints, VectorField2D, topography, VectorField
from .vector_calculus import (
    speed,
    jacobian,
    curl,
    divergence,
    acceleration,
    curvature,
    torsion,
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

# potential related
from .scPotential import (
    search_fixed_points,
    gen_fixed_points,
    gen_gradient,
    IntGrad,
    DiffusionMatrix,
    action,
    Potential,
)  # , vector_field_function
from .Bhattacharya import path_integral, alignment
from .Wang import Wang_action, Wang_LAP, transition_rate, MFPT
from .Ao import Ao_pot_map, solveQ

# state graph related

# dimension reduction related
from .dimension_reduction import reduceDimension  # , run_umap

# clustering related
from .clustering import hdbscan, cluster_field

# mnn related
from .connectivity import mnn, neighbors

# Pseudotime related
from .DDRTree import DDRTree_py as DDRTree
from .DDRTree import cal_ncenter

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

# stochastic process related
from .stochastic_process import diffusionMatrix

# vfGraph operation related:
from .vfGraph import vfGraph
