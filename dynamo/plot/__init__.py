"""Mapping Vector Field of Single Cells
"""

from .preprocessing import show_fraction, phase_portrait, featureGenes
from .dynamics import plot_fitting
from .scVectorField import cell_wise_velocity, grid_velocity, stremline_plot, plot_LIC # , plot_LIC_gray
from .scPotential import show_landscape
from .scatters import scatters
from .utilities import quiver_autoscaler, velocity_on_grid
