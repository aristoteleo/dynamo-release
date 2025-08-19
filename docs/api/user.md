# User

Import dynamo as:

```
import dynamo as dyn
```

```{eval-rst}
.. currentmodule:: dynamo

```

## Data IO

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   read
   read_h5ad
   read_loom
   read_h5

```

## Preprocessing (pp)

_Preprocessor class and recipe functions_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pp.recipe_monocle
   pp.recipe_velocyto
   pp.Preprocessor
   pp.filter_cells_by_outliers
   pp.filter_cells_by_highly_variable_genes
   

```

_Basic preprocessing functions_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pp.filter_cells
   pp.filter_cells_by_outliers
   pp.filter_cells_by_highly_variable_genes
   pp.filter_cells_by_outliers
   pp.filter_genes_by_clusters
   pp.filter_genes_by_outliers
   pp.filter_genes_by_pattern
   pp.filter_genes
   pp.normalize_cell_expr_by_size_factors
   pp.scale
   pp.log1p
   pp.pca

```

_Gene selection_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pp.highest_frac_genes
   pp.select_genes_monocle
   pp.select_genes_by_pearson_residuals

```

## Tools (tl)

_kNN and moments of expressions_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.neighbors
   tl.mnn
   tl.moments

```

_Kinetic parameters and RNA/protein velocity_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.dynamics

```

_Labeling Velocity recipes_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.recipe_deg_data
   tl.recipe_kin_data
   tl.recipe_mix_kin_deg_data
   tl.recipe_one_shot_data
   tl.velocity_N


```

_Labeling Velocity recipes_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.reduceDimension
   tl.DDRTree
   tl.psl

```

_Clustering_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.hdbscan
   tl.leiden
   tl.louvain
   tl.scc

```

_Velocity projection_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.cell_velocities
   tl.confident_cell_velocities

```

_Velocity metrics_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.cell_wise_confidence
   tl.gene_wise_confidence
   tl.pseudotime_velocity

```

_Markov chain_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.generalized_diffusion_map
   tl.stationary_distribution
   tl.diffusion
   tl.expected_return_time

```

_Markers and differential expressions_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.moran_i
   tl.find_group_markers
   tl.two_groups_degs
   tl.top_n_markers
   tl.glm_degs

```

_Cell proliferation and apoptosis_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.score_cells
   tl.cell_growth_rate

```

_Converter and helper_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.converter
   tl.run_scvelo
   tl.run_velocyto
   tl.vlm_to_adata

```

## Vector field (vf)

_Vector field reconstruction_

:::{note}
    Vector field class is internally to vf.VectorField. See our vector field classes here: [vector field](https://dynamo-release.readthedocs.io/en/latest/Class.html#vector-field)
:::

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   vf.VectorField
   vf.SparseVFC
   vf.BaseVectorField
   vf.SvcVectorField
   vf.graphize_vecfld
   vf.vector_field_function

```

_Vector field topology_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   vf.cluster_field
   vf.topography
   vf.FixedPoints
   vf.assign_fixedpoints

```

_Beyond RNA velocity_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   vf.velocities
   vf.speed
   vf.jacobian
   vf.divergence
   vf.curl
   vf.acceleration
   vf.curvature
   vf.torsion
   vf.sensitivity

```

_Beyond velocity vector field_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   vf.cell_accelerations
   vf.cell_curvatures

```

_Vector field ranking_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   vf.rank_genes
   vf.rank_expression_genes
   vf.rank_velocity_genes
   vf.rank_divergence_genes
   vf.rank_acceleration_genes
   vf.rank_curvature_genes
   vf.rank_jacobian_genes
   vf.rank_s_divergence_genes
   vf.rank_sensitivity_genes                

```

_Single cell potential: three approaches_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   vf.gen_fixed_points
   vf.gen_gradient
   vf.IntGrad
   vf.DiffusionMatrix
   vf.action
   vf.Potential
   vf.path_integral
   vf.alignment
   vf.Wang_action
   vf.Wang_LAP
   vf.transition_rate
   vf.MFPT
   vf.Ao_pot_map
   vf.solveQ          

```

_Stochastic processes_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:


   vf.diffusionMatrix             

```

_Vector field clustering and graph_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   vf.cluster_field
   vf.streamline_clusters
   vf.vfGraph         

```

## Prediction (pd)

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pd.andecestor
   pd.fate
   pd.fate_bias
   pd.get_init_path
   pd.least_action
   pd.perturbation
   pd.state_graph
   pd.KO
   pd.rank_perturbation_cell_clusters
   pd.rank_perturbation_cells
   pd.rank_perturbation_genes
   pd.tree_model     

```

## Plotting (pl)

_Preprocessing_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.basic_stats
   pl.show_fraction
   pl.feature_genes
   pl.biplot
   pl.loading
   pl.variance_explained
   pl.highest_frac_genes
   pl.exp_by_groups
   pl.bubble

```

_Cell cycle staging_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.cell_cycle_scores

```

_Scatter base_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.scatters

```

_Space plot_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.space

```

_Phase diagram: conventional scRNA-seq_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.phase_portraits

```

_Kinetic models: labeling based scRNA-seq_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.dynamics

```

_Kinetics_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.kinetic_curves
   pl.kinetic_heatmap
   pl.jacobian_kinetics
   pl.sensitivity_kinetics

```

_Dimension reduction_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.pca
   pl.tsne
   pl.umap
   pl.trimap

```

_Clustering_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.leiden
   pl.louvain
   pl.infomap
   pl.streamline_clusters

```

_Neighbor graph_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.nneighbors
   pl.state_graph
   
```

_Vector field plots: velocities and accelerations_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.cell_wise_vectors
   pl.cell_wise_vectors_3d
   pl.grid_vectors
   pl.streamline_plot
   pl.line_integral_conv
   pl.plot_energy
   pl.plot_3d_streamtube
   
```

_Vector field topology_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.plot_flow_field
   pl.plot_fixed_points
   pl.plot_fixed_points_2d
   pl.plot_nullclines
   pl.plot_separatrix
   pl.plot_traj
   pl.topography
   pl.response
   
```

_Beyond RNA velocity_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.speed
   pl.divergence
   pl.acceleration
   pl.curl
   pl.curvature
   pl.jacobian
   pl.jacobian_heatmap
   pl.sensitivity
   pl.sensitivity_heatmap
   
```

_Regulatory network_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.arcPlot
   pl.circosPlot
   pl.circosPlotDeprecated
   pl.hivePlot
   
```

_Potential landscape_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.show_landscape
   
```

_Cell fate_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.fate
   pl.fate_bias
   
```

_Heatmaps_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.causality
   pl.comb_logic
   pl.plot_hill_function
   pl.response
   
```

_Predictions_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.lap_min_time
   
```

_Save figures_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.save_fig
   
```

## Movie (mv)

:::{note}
    animation class is internally to mv.animate_fates. See our animation classes here: [animation](https://dynamo-release.readthedocs.io/en/latest/Class.html#movie)
:::

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   mv.animate_fates
   
``` 

## Simulation (sim)

_Simple ODE vector field simulation_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   sim.toggle
   sim.Ying_model
   
``` 

_Gillespie simulation_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   sim.Gillespie
   sim.Simulator
   sim.state_space_sampler
   sim.evaluate
   
``` 

## External (ext)

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   ext.ddhodge
   ext.enrichr
   ext.scribe
   ext.coexp_measure
   ext.scifate_glmnet
   
``` 

## Utilities

_Package versions_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   get_all_dependencies_version
   
``` 

_Clean up adata_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   cleanup
   
``` 

_Figures configuration_

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   configuration.set_figure_params
   configuration.set_pub_style
   
``` 

[anndata]: https://anndata.readthedocs.io/en/stable/
[scanpy]: https://scanpy.readthedocs.io/en/stable/index.html
[utilities]: https://scanpy.readthedocs.io/en/stable/api/index.html#reading
[ray tune]: https://docs.ray.io/en/latest/tune/index.html
