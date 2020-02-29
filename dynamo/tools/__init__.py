"""Mapping Vector Field of Single Cells
"""

# inclusive expression dynamics model related 
# from .dynamo import sol_u, sol_s, fit_gamma_labelling, fit_alpha_labelling, fit_gamma_splicing, fit_gamma
# from .dynamo_fitting import sol_u, sol_s, sol_p, sol_ode, sol_num, fit_gamma_labelling, fit_beta_lsq, fit_alpha_labelling, fit_alpha_synthesis, fit_gamma_splicing, fit_gamma
from .moments import Estimation

from .velocity import sol_u, sol_s, sol_p, fit_linreg, fit_first_order_deg_lsq, solve_first_order_deg, fit_gamma_lsq, fit_alpha_synthesis, fit_alpha_degradation, velocity, estimation
from .cell_velocities import cell_velocities, generalized_diffusion_map, stationary_distribution, diffusion, expected_return_time

from .dynamics import dynamics

# run other velocity tools: 
from .velocyto_scvelo import vlm_to_adata, converter, run_velocyto, run_scvelo, mean_var_by_time, run_dynamo, run_dynamo_simple_fit, run_dynamo_labelling, compare_res

# vector field related
from .velocity_metric import cell_wise_confidence 
from .scVectorField import SparseVFC, con_K, get_P, VectorField #, evaluate, con_K_div_cur_free, vector_field_function, vector_field_function_auto, auto_con_K
from .topography import FixedPoints, VectorField2D, topography

# Markov chain related:
from .Markov import markov_combination, compute_markov_trans_prob, compute_kernel_trans_prob, compute_drift_kernel, compute_drift_local_kernel, compute_density_kernel, makeTransitionMatrix, compute_tau, smoothen_drift_on_grid, MarkovChain, KernelMarkovChain, DiscreteTimeMarkovChain, ContinuousTimeMarkovChain

# potential related
from .scPotential import gen_fixed_points, gen_gradient, IntGrad, DiffusionMatrix, action, Potential #, vector_field_function
from .Bhattacharya import path_integral, alignment
from .Wang import Wang_action, Wang_LAP, transition_rate, MFPT
from .Ao import Ao_pot_map

# cell fate related
from .fate import Fate, fate

# dimension reduction related
from .dimension_reduction import reduceDimension #, run_umap

# mnn related
from .connectivity import mnn, smoother

# Pseudotime related
from .DDRTree import DDRTree_py as DDRTree
from .DDRTree import cal_ncenter

# Utils
from .utils import vector_field_function

