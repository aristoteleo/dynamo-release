.. automodule:: dynamo

API
===

Import dynamo as::

   import dynamo as dyn

Data IO
~~~~~~~
(See more at `anndata-docs <https://anndata.readthedocs.io/en/latest/anndata.AnnData.html>`_)

.. autosummary::
   :toctree: _autosummary

   read
   read_h5ad
   read_loom


Preprocessing (pp)
~~~~~~~~~~~~~~~~~~

.. autosummary::
      :toctree: _autosummary

   pp.recipe_monocle
   pp.cell_cycle_scores
   .. pp.pca


Estimation (est)
~~~~~~~~~~~~~~~~

.. note::
   Classes in **est** are internally to **Tools**. See our estimation classes here: `estimation`_


Tools (tl)
~~~~~~~~~~

*kNN and moments of expressions*

.. autosummary::
    :toctree: _autosummary

   tl.neighbors
   tl.mnn
   tl.moments

*Kinetic parameters and RNA/protein velocity*

.. autosummary::
    :toctree: _autosummary

   tl.dynamics

   .. tl.calc_1nd_moment
   .. tl.calc_2nd_moment

*Dimension reduction*

.. autosummary::
    :toctree: _autosummary

   tl.reduceDimension
   tl.DDRTree
   tl.psl
   .. tl.cal_ncenter

*Clustering*

.. autosummary::
    :toctree: _autosummary

   tl.hdbscan
   tl.cluster_field

*Velocity projection*

.. autosummary::
    :toctree: _autosummary

   tl.cell_velocities
   tl.confident_cell_velocities

*Velocity metrics*

.. autosummary::
    :toctree: _autosummary

   tl.cell_wise_confidence
   tl.gene_wise_confidence

*Markov chain*

.. autosummary::
    :toctree: _autosummary

   tl.generalized_diffusion_map
   tl.stationary_distribution
   tl.diffusion
   tl.expected_return_time

   .. tl.markov_combination
   .. tl.compute_markov_trans_prob
   .. tl.compute_kernel_trans_prob
   .. tl.compute_drift_kernel
   .. tl.compute_drift_local_kernel
   .. tl.compute_density_kernel
   .. tl.makeTransitionMatrix
   .. tl.compute_tau
   .. tl.MarkovChain
   .. tl.KernelMarkovChain
   .. tl.DiscreteTimeMarkovChain
   .. tl.ContinuousTimeMarkovChain

*Markers and differential expressions*

.. autosummary::
    :toctree: _autosummary

   tl.moran_i
   tl.find_group_markers
   tl.two_groups_degs
   tl.top_n_markers
   tl.glm_degs

   .. tl.TRNET
   .. tl.trn
   .. tl.sample_by_velocity
   .. tl.lhsclassic
   .. tl.sample

*Cell proliferation and apoptosis*

.. autosummary::
    :toctree: _autosummary

   tl.score_cells
   tl.cell_growth_rate

   .. tl.n_descendants
   .. tl.growth_rate

*Converter*

.. autosummary::
    :toctree: _autosummary

   tl.converter

Vector field (vf)
~~~~~~~~~~~~~~~~~

*Vector field reconstruction*

.. note::
   Vector field class is internally to `vf.VectorField`. See our vector field classes here: `vector field`_

.. autosummary::
    :toctree: _autosummary

   vf.SparseVFC
   vf.base_vectorfield
   vf.SvcVectorfield
   .. vf.graphize_vecfld
   .. vf.vector_field_function


*Vector field topology*

.. autosummary::
    :toctree: _autosummary

   vf.topography
   .. vf.FixedPoints
   .. vf.VectorField2D

*Beyond RNA velocity*

.. autosummary::
    :toctree: _autosummary

   vf.velocities
   vf.speed
   vf.divergence
   vf.curl
   vf.acceleration
   vf.curvature
   vf.torsion

*Beyond velocity vector field*

.. autosummary::
    :toctree: _autosummary

   vf.cell_accelerations
   vf.cell_curvatures

*Vector field ranking*

.. autosummary::
    :toctree: _autosummary

   vf.rank_velocity_genes
   vf.rank_divergence_genes
   vf.rank_acceleration_genes
   vf.rank_curvature_genes

*Single cell potential: three approaches*

.. autosummary::
    :toctree: _autosummary

   vf.gen_fixed_points
   vf.gen_gradient
   vf.IntGrad
   vf.DiffusionMatrix
   vf.action
   vf.Potential
   .. vf.search_fixed_points

   vf.path_integral
   vf.alignment
   vf.Wang_action
   vf.Wang_LAP
   vf.transition_rate
   vf.MFPT

   vf.Ao_pot_map
   vf.solveQ


*Stochastic processes*

.. autosummary::
    :toctree: _autosummary

   vf.diffusionMatrix


*Vector field graph*

.. autosummary::
    :toctree: _autosummary

   vf.vfGraph

Prediction (pd)
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   pd.fate
   pd.fate_bias
   pd.state_graph

Plotting (pl)
~~~~~~~~~~~~~


*Preprocessing*

.. autosummary::
    :toctree: _autosummary

   pl.basic_stats
   pl.show_fraction
   pl.feature_genes
   pl.variance_explained
   pl.exp_by_groups


*Cell cycle staging*

.. autosummary::
    :toctree: _autosummary

   pl.cell_cycle_scores


*Scatter base*

.. autosummary::
    :toctree: _autosummary

   pl.scatters


*Phase diagram: conventional scRNA-seq*

.. autosummary::
    :toctree: _autosummary

   pl.phase_portraits


*Kinetic models: labeling based scRNA-seq*

.. autosummary::
    :toctree: _autosummary

   pl.dynamics


*Kinetics*

.. autosummary::
    :toctree: _autosummary

   pl.kinetic_curves
   pl.kinetic_heatmap
   pl.jacobian_kinetics


*Dimension reduction*

.. autosummary::
    :toctree: _autosummary

   pl.pca
   pl.tsne
   pl.umap
   pl.trimap


*Neighbor graph*

.. autosummary::
    :toctree: _autosummary

   pl.nneighbors
   pl.state_graph


*Vector field plots: velocities and accelerations*

.. autosummary::
    :toctree: _autosummary

   pl.cell_wise_vectors
   pl.grid_vectors
   pl.streamline_plot
   pl.line_integral_conv
   pl.plot_energy


*Vector field topology*

.. autosummary::
    :toctree: _autosummary

   pl.plot_flow_field
   pl.plot_fixed_points
   pl.plot_nullclines
   pl.plot_separatrix
   pl.plot_traj
   pl.topography


*Beyond RNA velocity*

.. autosummary::
    :toctree: _autosummary

   pl.speed
   pl.divergence
   pl.curl
   pl.curvature
   pl.jacobian
   pl.jacobian_heatmap


*Potential landscape*

.. autosummary::
    :toctree: _autosummary

   pl.show_landscape


*Cell fate*

.. autosummary::
    :toctree: _autosummary

   pl.fate_bias


*Save figures*

.. autosummary::
    :toctree: _autosummary

   pl.save_fig

Moive (mv)
~~~~~~~~~~~~~

.. note::
   animation class is internally to `mv.animate_fates`. See our animation classes here: `animation`_

.. autosummary::
   :toctree: _autosummary

   mv.animate_fates

Simulation (sim)
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary


*Simple ODE vector field simulation*

.. autosummary::
    :toctree: _autosummary

   sim.two_genes_motif
   sim.neurogenesis
   sim.toggle
   sim.Ying_model


*Gillespie simulation*

.. autosummary::
    :toctree: _autosummary

   sim.Gillespie
   sim.Simulator

   sim.state_space_sampler
   sim.evaluate

External (ext)
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

    ext.ddhodge
    ext.enrichr
    ext.scribe
    ext.coexp_measure
    ext.scifate_glmnet

Utilities
~~~~~~~~~

*Package versions*

.. autosummary::
   :toctree: _autosummary

    get_all_dependencies_version

*Clean up adata*

.. autosummary::
   :toctree: _autosummary

    cleanup

*Figures configuration*

.. autosummary::
   :toctree: _autosummary

   configuration.set_figure_params
   configuration.set_pub_style

.. _`estimation`: ./Class.html#estimation
.. _`vector field`: ./Class.html#vector-field
.. _`animation`: ./Class.html#movie
