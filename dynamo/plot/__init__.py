"""Mapping Vector Field of Single Cells
"""

from .theme import points
from .preprocess import show_fraction, feature_genes, variance_explained
from .scatters import scatters
from .dynamics import phase_portraits, dynamics
from .scVectorField import cell_wise_velocity, grid_velocity, stremline_plot, line_integral_conv # , plot_LIC_gray
from .scPotential import show_landscape
from .connectivity import nneighbors
from .time_series import kinetic_heatmap, kinetic_curves
from .topology import plot_flow_field, plot_fixed_points, plot_null_clines, plot_separatrix, plot_traj, topography
from .utils import quiver_autoscaler
