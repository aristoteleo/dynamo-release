"""Mapping Vector Field of Single Cells
"""

from .preprocessing import show_fraction, phase_portrait, featureGenes, variance_explained
from .dynamics import plot_fitting
from .scVectorField import cell_wise_velocity, grid_velocity, stremline_plot, line_integral_conv # , plot_LIC_gray
from .scPotential import show_landscape
from .scatters import scatters
from .time_series import kinetic_heatmap, kinetic_curves, plot_directed_pg
from .utilities import quiver_autoscaler
