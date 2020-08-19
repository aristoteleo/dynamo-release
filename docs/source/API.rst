.. automodule:: dynamo

API
===

Import dynamo as::

   import dynamo as dyn

Preprocessing (pp)
~~~~~~~~~~~~~~~~~~

.. autosummary::
      :toctree: _autosummary

   pp.recipe_monocle
   pp.cell_cycle_scores
   .. pp.pca


Estimation (est.csc)
~~~~~~~~~~~~~~~~~~~~

**Conventional scRNA-seq (est.csc)**

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   est.csc.velocity
   est.csc.ss_estimation

**Time-resolved metabolic labeling based scRNA-seq (est.tsc)**

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   est.tsc.kinetic_estimation
   est.tsc.Estimation_DeterministicDeg
   est.tsc.Estimation_DeterministicDegNosp
   est.tsc.Estimation_MomentDeg
   est.tsc.Estimation_MomentDegNosp
   est.tsc.Estimation_MomentKin
   est.tsc.Estimation_MomentKinNosp
   est.tsc.Estimation_DeterministicKinNosp
   est.tsc.Estimation_DeterministicKin
   est.tsc.Mixture_KinDeg_NoSwitching
   est.tsc.Lambda_NoSwitching


Tools (tl)
~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

   tl.neighbors
   tl.mnn
   tl.moments
   tl.dynamics

   .. tl.calc_1nd_moment
   .. tl.calc_2nd_moment

   tl.reduceDimension

   tl.hdbscan
   tl.cluster_field

   tl.cell_velocities
   tl.confident_cell_velocities

   tl.cell_wise_confidence
   tl.gene_wise_confidence

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

   tl.converter

   tl.DDRTree
   tl.psl
   .. tl.cal_ncenter

Vector field (vf)
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   vf.SparseVFC
   vf.VectorField
   .. vf.graphize_vecfld
   .. vf.vector_field_function

   vf.topography
   .. vf.FixedPoints
   .. vf.VectorField2D

   vf.cell_accelerations

   vf.speed
   vf.divergence
   vf.curl
   vf.acceleration
   vf.curvature
   vf.torsion
   vf.Potential

   vf.rank_speed_genes
   vf.rank_divergence_genes
   vf.rank_acceleration_genes
   vf.rank_curvature_genes

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

   vf.diffusionMatrix

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

.. autosummary::
   :toctree: _autosummary

   pl.basic_stats
   pl.show_fraction
   pl.feature_genes
   pl.variance_explained
   pl.exp_by_groups

   pl.cell_cycle_scores

   pl.scatters
   pl.phase_portraits
   pl.dynamics

   pl.kinetic_curves
   pl.kinetic_heatmap
   pl.jacobian_kinetics

   pl.pca
   pl.tsne
   pl.umap
   pl.trimap

   pl.nneighbors
   pl.state_graph

   pl.cell_wise_vectors
   pl.grid_vectors
   pl.streamline_plot
   pl.line_integral_conv
   pl.plot_energy

   pl.plot_flow_field
   pl.plot_fixed_points
   pl.plot_nullclines
   pl.plot_separatrix
   pl.plot_traj
   pl.topography

   pl.speed
   pl.divergence
   pl.curl
   pl.curvature
   pl.jacobian
   pl.jacobian_heatmap

   pl.show_landscape

   pl.fate_bias

   pl.save_fig

Moive (mv)
~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   mv.StreamFuncAnim
   mv.animate_fates

Simulation (sim)
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   sim.two_genes_motif
   sim.neurogenesis
   sim.toggle
   sim.Ying_model

   sim.Gillespie
   sim.Simulator

   sim.state_space_sampler
   sim.evaluate

External (ext)
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

    ext.ddhodge
    ext.scribe
    ext.mutual_inform
    ext.scifate_glmnet
