"""Mapping Vector Field of Single Cells
"""

from .preprocess import show_fraction, feature_genes, variance_explained
from .dynamics import phase_portraits, dynamics
from .scVectorField import cell_wise_velocity, grid_velocity, stremline_plot, line_integral_conv # , plot_LIC_gray
from .scPotential import show_landscape
from .scatters import scatters
from .time_series import kinetic_heatmap, kinetic_curves, plot_directed_pg
from .utilities import quiver_autoscaler
from .topology import plot_flow_field, plot_fixed_points, plot_null_clines, plot_separatrix, plot_traj, topography
