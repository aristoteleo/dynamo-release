# Release Notes

## Dynamo Ver 1.4.1

### DEBUG

- Debug and refactor scPotential ([PR 624](https://github.com/aristoteleo/dynamo-release/pull/624)).
- Replace deprecated `np.asscalar()` with `np.ndarray.item()` ([PR 643](https://github.com/aristoteleo/dynamo-release/pull/643)).
- Create chunk option for normalization and gene selection ([PR 598](https://github.com/aristoteleo/dynamo-release/pull/598)).
- Debug `pd.state_graph()` ([PR 630](https://github.com/aristoteleo/dynamo-release/pull/630)).
- Debug `pl.jacobian_heatmap()` ([PR 653](https://github.com/aristoteleo/dynamo-release/pull/653)).
- Debug `pl.nneighbors()` ([PR 644](https://github.com/aristoteleo/dynamo-release/pull/644)).
- Retry codecov upload ([PR 656](https://github.com/aristoteleo/dynamo-release/pull/656)).
- Debug vectorfield given layer input ([PR 619](https://github.com/aristoteleo/dynamo-release/pull/619)).
- Debug simulation module ([PR 658](https://github.com/aristoteleo/dynamo-release/pull/658)).
- Extra filter after pearson residuals normalization ([PR 665](https://github.com/aristoteleo/dynamo-release/pull/665)).
- Add missing return value to deprecated functions ([PR 663](https://github.com/aristoteleo/dynamo-release/pull/663)).
- Debug networks plot ([PR 657](https://github.com/aristoteleo/dynamo-release/pull/657)).
- Implement `pl.plot_connectivity()` ([PR 652](https://github.com/aristoteleo/dynamo-release/pull/652)).
- Debug the preprocessing of integer matrix input ([PR 664](https://github.com/aristoteleo/dynamo-release/pull/664)).
- Missing return value in `pl.lap_min_time()` ([PR 668](https://github.com/aristoteleo/dynamo-release/pull/668)).
- Update matplotlib `Colorbar.draw_all()` to `Colorbar._draw_all()` ([PR 669](https://github.com/aristoteleo/dynamo-release/pull/669)).
- Optimize code coverage tests ([PR 605](https://github.com/aristoteleo/dynamo-release/pull/605)).
- Debug `test_gradop()` by ([PR 677](https://github.com/aristoteleo/dynamo-release/pull/677)).
- Constraint on matplotlib version by ([PR 679](https://github.com/aristoteleo/dynamo-release/pull/679)).
- Upgrade code coverage to v4 ([PR 684](https://github.com/aristoteleo/dynamo-release/pull/684)).
- Init a branch for updating dependency ([PR 690](https://github.com/aristoteleo/dynamo-release/pull/690)).
- Replace `louvain` with `leiden` ([PR 692](https://github.com/aristoteleo/dynamo-release/pull/692)).
- Debug `pl.highest_frac_genes()` ([PR 681](https://github.com/aristoteleo/dynamo-release/pull/681)).
- Deprecate more sparse matrix `.A` attributes ([PR 695](https://github.com/aristoteleo/dynamo-release/pull/695)).
- Fix matplotlib version issues and a circular import issue ([PR 686](https://github.com/aristoteleo/dynamo-release/pull/686)).
- Debug `set_figure_params()` ([PR 698](https://github.com/aristoteleo/dynamo-release/pull/698)).
- Debug: shape and name mismatch in cell-wise alpha saving ([PR 697](https://github.com/aristoteleo/dynamo-release/pull/697)).
- Debug: The sizes of the scatter plots are not set correctly ([PR 696](https://github.com/aristoteleo/dynamo-release/pull/696)).

### Others

- Refactor `pd.fate()` with Trajectory class ([PR 645](https://github.com/aristoteleo/dynamo-release/pull/645)).
- Reorganize estimation module ([PR 662](https://github.com/aristoteleo/dynamo-release/pull/662)).
- Refactor `pl.scatters()` and `pl.scatters_interactive()` ([PR 654](https://github.com/aristoteleo/dynamo-release/pull/654)).
- Refactor `vf.VectorField()` function ([PR 620](https://github.com/aristoteleo/dynamo-release/pull/620)).
- Docstring and type hints for the prediction module ([PR 666](https://github.com/aristoteleo/dynamo-release/pull/666)).
- Update docstr and type hints for External module ([PR 661](https://github.com/aristoteleo/dynamo-release/pull/661)).
- Add doctring and type hints for simulation module ([PR 660](https://github.com/aristoteleo/dynamo-release/pull/660)).
- Docstring and type hints for root folder python files ([PR 667](https://github.com/aristoteleo/dynamo-release/pull/667)).

## Dynamo Ver 1.4.0

### Feature Changes

- Shiny web application for in silico perturbation and least square action path analyses
  ([PR 582](https://github.com/aristoteleo/dynamo-release/pull/582)).

- More 3D plots ([PR 597](https://github.com/aristoteleo/dynamo-release/pull/597)):

  - 3D scatters with Plotly and Pyvista `dyn.pl.scatters_interactive()`.
  - 3D vectors with Plotly and Pyvista `dyn.pl.cell_wise_vectors_3d()`.
  - 3D topography with Plotly and Pyvista `dyn.pl.topography_3d()`.
  - 3D animation with Pyvista `dyn.mv.PyvistaAnim()`.

- Saved the velocity parameters in `adata.varm` instead of `adata.var`
  ([PR 579](https://github.com/aristoteleo/dynamo-release/pull/579)).

- DDRtree based pseudotime and graph learning ([PR 564](https://github.com/aristoteleo/dynamo-release/pull/564)):
  `dyn.tl.order_cells()`, `dyn.tl.construct_velocity_tree()`.

- Integrated `hnswlib` fast nearest neighbors method ([PR 552](https://github.com/aristoteleo/dynamo-release/pull/552)).

- A helper functon to convert the AnnData object from Dynamo to Scvelo, or vice versa
  ([PR 551](https://github.com/aristoteleo/dynamo-release/pull/551)).

- The tools module has been reorganized ([PR 625](https://github.com/aristoteleo/dynamo-release/pull/625)):

  - Deprecate files `dynamo_fitting.py`, `dynamo_bk.py`, `dynamics_deprecated.py`, `utils_moments_deprecated.py`.
  - Deprecate legacy functions in `construct_velocity_tree.py`,`pseudotime.py`, `moments.py`, `clustering.py`.
  - Merge `utils_markers.py` and `markers.py`.
  - Merge `time_series.py` (learns a direct principal graph by integrating the transition matrix between and DDRTree)
    and `construct_velocity_tree.py`(Integrate pseudotime ordering with velocity to automatically assign the direction
    of the learned trajectory.) to `DDRTree_graph.py`.
  - Reorganize some functions to utils in the following file: `time_series.py`, `multiomics.py`.
  - Rename: `DDRTree_py.py` to `DDRTree.py`, `psl_py.py` to `psl.py`.

- Deprecate infomap clustering ([PR 555](https://github.com/aristoteleo/dynamo-release/pull/555)).

### DEBUG

- Fixed the bug that the `dyn.pl.kinetic_heatmap()` couldn't be transposed caused by wrong initialization
  ([PR 558](https://github.com/aristoteleo/dynamo-release/pull/558))
  ([PR 636](https://github.com/aristoteleo/dynamo-release/pull/636)).
- Fixed the bug that `dyn.pl.cell_wise_vectors()` only output one color
  ([PR 559](https://github.com/aristoteleo/dynamo-release/pull/559)).
- Debugged the sampling method in tools modules
  ([PR 565](https://github.com/aristoteleo/dynamo-release/pull/565)).
- Fixed the panda error in `dyn.tl.gene_wise_confidence()`
  ([PR 567](https://github.com/aristoteleo/dynamo-release/pull/567)).
- Fixed the bug that `pysal` submodules were not imported explicitly
  ([PR 568](https://github.com/aristoteleo/dynamo-release/pull/568)).
- Debugged `dyn.tl.score_cells()` ([PR 569](https://github.com/aristoteleo/dynamo-release/pull/569)).
- Debugged the ambiguous if statement in `dyn.tl.psl()`
  ([PR 573](https://github.com/aristoteleo/dynamo-release/pull/573)).
- Updated all the expired links of sample dataset ([PR 577](https://github.com/aristoteleo/dynamo-release/pull/577)).
- Fixed the bug that processed AnnData object couldn't be saved under some cases
  ([PR 580](https://github.com/aristoteleo/dynamo-release/pull/580)).
- Debugged `pp/transform.py` ([PR 581](https://github.com/aristoteleo/dynamo-release/pull/581)).
- Debugged `dyn.tl.cell_velocities()` ([PR 585](https://github.com/aristoteleo/dynamo-release/pull/585)).
- Debugged `dyn.pl.kinetic_curves()` ([PR 587](https://github.com/aristoteleo/dynamo-release/pull/587)).
- Fixed the error caused by wrong type hints in `dyn.tl.BaseVectorField.find_fixed_points()`
  ([PR 597](https://github.com/aristoteleo/dynamo-release/pull/597)).
- Fixed the error caused by excessive memory usage in tests
  ([PR 602](https://github.com/aristoteleo/dynamo-release/pull/602)).
- Fixed the KeyError in `dyn.pp.convert2symbol()` when all genes are found
  ([PR 603](https://github.com/aristoteleo/dynamo-release/pull/603)).
- Fixed the issue that `dyn.pp.highest_frac_genes()` didn't support sparse input
  ([PR 604](https://github.com/aristoteleo/dynamo-release/pull/604)).
- Debugged `dyn.tl.cell_growth_rate()` ([PR 606](https://github.com/aristoteleo/dynamo-release/pull/606)).
- Debugged the arclength sampling method in `dyn.pd.fate()`
  ([PR 592](https://github.com/aristoteleo/dynamo-release/pull/592))
  ([PR 610](https://github.com/aristoteleo/dynamo-release/pull/610)).
- Removed unnecessary import of pandas ([PR 614](https://github.com/aristoteleo/dynamo-release/pull/614)).
- Debugged the `dyn.pl.topography()` when the color is not provided
  ([PR 617](https://github.com/aristoteleo/dynamo-release/pull/617)).
- Fixed the error that list object doesn't have to_list() method in `dyn.vf.hessian()`
  ([PR 623](https://github.com/aristoteleo/dynamo-release/pull/623)).
- Fixed the ambiguous if statement in the `dyn.tl.MarkovChain.is_normalized()`
  ([PR 626](https://github.com/aristoteleo/dynamo-release/pull/626)).
- Debugged the `dyn.pd.classify_clone_cell_type()` ([PR 627](https://github.com/aristoteleo/dynamo-release/pull/627)).
- Fixed the input of `minimize()` in `dyn.pd.lap_T()`
  ([PR 628](https://github.com/aristoteleo/dynamo-release/pull/628)).
- Fixed the bug that average parameter didn't work in `dyn.pd.fate()`
  ([PR 629](https://github.com/aristoteleo/dynamo-release/pull/629)).
- Debugged the `dyn.pl.line_integral_conv()` ([PR 639](https://github.com/aristoteleo/dynamo-release/pull/639)).

### Others

- Now available on [conda forge](https://anaconda.org/conda-forge/dynamo-release).
- Removed `cdlib` dependency ([PR 532](https://github.com/aristoteleo/dynamo-release/pull/532)).
- Removed `KDEpy` dependency ([PR 533](https://github.com/aristoteleo/dynamo-release/pull/533)).
- Added code coverage report ([PR 555](https://github.com/aristoteleo/dynamo-release/pull/555)).
- Optimized the structure of the umap dimension reduction
  ([PR 556](https://github.com/aristoteleo/dynamo-release/pull/556)).
- Optimized the structure and supported sparse input in `tools/graph_calculus.py`
  ([PR 557](https://github.com/aristoteleo/dynamo-release/pull/557)).
- Updated `networkx` API ([PR 560](https://github.com/aristoteleo/dynamo-release/pull/560)).
- Replaced `python-igraph` dependency with `igraph` ([PR 563](https://github.com/aristoteleo/dynamo-release/pull/563)).
- Added docstrings for tools module ([PR 570](https://github.com/aristoteleo/dynamo-release/pull/570)).
- Removed duplicate size factor calculation ([PR 596](https://github.com/aristoteleo/dynamo-release/pull/596)).
- Implemented a helper function for saving the plots
  ([PR 609](https://github.com/aristoteleo/dynamo-release/pull/609))
  ([PR 635](https://github.com/aristoteleo/dynamo-release/pull/635)).
- Added docstrings for estimation module ([PR 611](https://github.com/aristoteleo/dynamo-release/pull/611)).
- Merged `dyn.pd.rank_cells()` and `dyn.pd.rank_cell_groups()`
  ([PR 613](https://github.com/aristoteleo/dynamo-release/pull/613)).
- Added the conda badge ([PR 618](https://github.com/aristoteleo/dynamo-release/pull/618)).
- Handled the duplicate files when downloading sample data
  ([PR 621](https://github.com/aristoteleo/dynamo-release/pull/621)).
- Debugged the ROC curve in Shiny app ([PR 637](https://github.com/aristoteleo/dynamo-release/pull/637)).

## Dynamo Ver 1.3.0

### Feature Changes

- The preprocessing module has been refactored:

  - Class *Preprocessor* is recommended for most preprocessing methods and recipes. `pp.recipe_monocle,`
    `pp.recipe_velocyto` has been deprecated ([PR 497](https://github.com/aristoteleo/dynamo-release/pull/497))
    ([PR 500](https://github.com/aristoteleo/dynamo-release/pull/500)).
    Check the tutorials [here](Preprocessor_tutorial.rst) for more instructions.
  - Normalization has been refactored ([PR 474](https://github.com/aristoteleo/dynamo-release/pull/474))
    ([PR 475](https://github.com/aristoteleo/dynamo-release/pull/475)): `pp.normalize_cell_expr_by_size_factors`
    has been deprecated, and new APIs are:

    - `pp.normalize_cell_expr_by_size_factors` -> `pp.calc_sz_factor, pp.normalize`.

  - Gene selection has been refactored ([PR 474](https://github.com/aristoteleo/dynamo-release/pull/474)). Now support
    genes selected by fano factors. APIs are `pp.select_genes_monocle` and `pp.select_genes_by_seurat_recipe`.
  - PCA has been refactored ([PR 469](https://github.com/aristoteleo/dynamo-release/pull/469)). `dyn.pp.pca_monocle`
    has been deprecated. The new API is:

    - `pp.pca_monocle` -> `pp.pca`.

  - sctransform and pearson residuals recipe has been refactored
    ([PR 510](https://github.com/aristoteleo/dynamo-release/pull/510))
    ([PR 512](https://github.com/aristoteleo/dynamo-release/pull/512)). Now those advanced methods will only be
    performed on X layer. Other layers will get normalized by size factors.
  - Calculation of `ntr` rate and `pp.cell_cycle_scores` has been added to the Preprocessor
    ([PR 513](https://github.com/aristoteleo/dynamo-release/pull/513)). To enable cell cycle scores, set parameter
    `cell_cycle_score_enable` to `True` when initializing the `pp.Preprocessor`.
  - Now the size factors normalization will normalize all layers with its own size factors by default
    ([PR 521](https://github.com/aristoteleo/dynamo-release/pull/521)). To normalize the labeled data with total size
    factors, we need to set the `total_szfactor` to `total_Size_Factor` explicitly.
  - Multiple new features added, includes genes selection by fano factors
    ([PR 474](https://github.com/aristoteleo/dynamo-release/pull/474)), external data integration methods
    ([PR 473](https://github.com/aristoteleo/dynamo-release/pull/473)) and `pp.regress_out`
    ([PR 470](https://github.com/aristoteleo/dynamo-release/pull/470))
    ([PR 483](https://github.com/aristoteleo/dynamo-release/pull/483))
    ([PR 484](https://github.com/aristoteleo/dynamo-release/pull/484)).
  - Created more tests for preprocessing module ([PR 485](https://github.com/aristoteleo/dynamo-release/pull/485)).
  - Replaced `adata.obsm["X"]` with `adata.obsm["X_pca"]`
    ([PR 514](https://github.com/aristoteleo/dynamo-release/pull/514)).
  - Removed some console output. They can still be displayed with `DEBUG` logging mode.
  - Other deprecated APIs include: `pp.calc_sz_factor_legacy, pp.filter_cells_legacy`,
    `pp.filter_genes_by_outliers_legacy, pp.select_genes_monocle_legacy, pp.select_genes_by_dispersion_general`,
    `pp.cook_dist, pp.normalize_cell_expr_by_size_factors`. More information can be found on our
    [preprocessing tutorials](Preprocessor_tutorial.rst).

### DEBUG

- Fixed the bug that save_show_or_return flags not working
  ([PR 414](https://github.com/aristoteleo/dynamo-release/pull/414)).
- Enabled the leiden algorithm to accept the resolution parameters
  ([PR 441](https://github.com/aristoteleo/dynamo-release/pull/441)).
- Fixed the wrong attribute name of anndata object in `utils_dimensionReduction.py`
  ([PR 458](https://github.com/aristoteleo/dynamo-release/pull/458)).
- Fixed the dimensionality issue in `moments.py`
  ([PR 461](https://github.com/aristoteleo/dynamo-release/pull/461)).
- Fixed part of the bug that h5ad file cannot be saved correctly
  ([PR 467](https://github.com/aristoteleo/dynamo-release/pull/467)).
- Fixed the bug that `pca_mean` will be `None` under some circumstances
  ([PR 482](https://github.com/aristoteleo/dynamo-release/pull/482)).
- Removing warning message for nxviz
  ([PR 489](https://github.com/aristoteleo/dynamo-release/pull/489)).
- Corrected the norm log-likelihood function
  ([PR 495](https://github.com/aristoteleo/dynamo-release/pull/495)).
- Removed deprecated parameters in gseapy functions
  ([PR 496](https://github.com/aristoteleo/dynamo-release/pull/496)).
- Fixed the bugs that functions will raise error when no fixed points are found in vector field by sampling
  ([PR 501](https://github.com/aristoteleo/dynamo-release/pull/501)).
- Removed unwanted operations in dimension reduction
  ([PR 502](https://github.com/aristoteleo/dynamo-release/pull/502)).

### Tutorial Updates on Readthedocs

- Documentation, Tutorials, and readthedocs update:

  - Update requirements for readthedocs ([PR 466](https://github.com/aristoteleo/dynamo-release/pull/466)).
  - Update readme ([PR 479](https://github.com/aristoteleo/dynamo-release/pull/479)).
  - Fixed documentation error caused by importing Literal
    ([PR 486](https://github.com/aristoteleo/dynamo-release/pull/486)).
  - Fixed readthedocs error caused by the new version of urllib3
    ([PR 488](https://github.com/aristoteleo/dynamo-release/pull/488)).

### Other Changes

- Docstring and type hints update:

  - Updated docstring and type hints for tools module
    ([PR 419](https://github.com/aristoteleo/dynamo-release/pull/419)).
  - Updated docstring and type hints for vector field module
    ([PR 434](https://github.com/aristoteleo/dynamo-release/pull/434)).
  - Updated the docstring and type hints for simulation and predicting module
    ([PR 457](https://github.com/aristoteleo/dynamo-release/pull/457)).
  - Update the docstring and type hints for hzplot
    ([PR 456](https://github.com/aristoteleo/dynamo-release/pull/456)).

## Dynamo Ver 1.1.0

### Feature Changes

- Following new function are added, exported or documented in API / class page: 
  
  - *Preprocessing*: `pp.convert2symbol, pp.filter_cells, pp.filter_gene,` 
    `pp.filter_genes_by_pattern, pp.normalize_cells, pp.scale, pp.log1p, pp.pca`
  - *Kinetic parameters and RNA/protein velocity*: `tl.recipe_deg_data, tl.recipe_kin_data,`
    `tl.recipe_mix_kin_deg_data, tl.recipe_one_shot_data, tl.velocity_N`
  - *Labeling Velocity recipes*: `tl.infomap, tl.leiden, tl.louvain, tl.scc`
  - *Clustering*: `tl.run_scvelo, tl.run_velocyto, tl.vlm_to_adata`
  - *Converter and helper*: `vf.graphize_vecfld, vf.vector_field_function`
  - *Vector field reconstruction*: `vf.FixedPoints, vf.VectorField2D, vf.assign_fixedpoints`
  - *Beyond RNA velocity*: `vf.jacobian, vf.sensitivity`
  - *Vector field ranking*: `vf.rank_cells, vf.rank_genes, vf.rank_expression_genes,`
    `vf.rank_jacobian_genes, vf.rank_s_divergence_genes, vf.rank_sensitivity_genes`
  - *Vector field clustering and graph*: `vf.cluster_field, vf.streamline_clusters`
  - *Prediction* `pd.andecestor, pd.get_init_path, pd.least_action, pd.perturbation,`
    `pd.rank_perturbation_cell_clusters, pd.rank_perturbation_cells, pd.rank_perturbation_genes,`
    `pd.state_graph, pd.tree_model`
  - *Preprocessing plot*: `pl.biplot, pl.loading, pl.highest_frac_genes, pl.bubble`
  - *Space plot*: `pl.space`
  - *Kinetics plot*: `pl.sensitivity_kinetics`
  - *Vector field plots*: `pl.cell_wise_vectors_3d, pl.plot_fixed_points_2d`
  - *differential geometry plots*: `pl.acceleration`
  - *Regulatory network plots* `pl.arcPlot, pl.circosPlot, pl.circosPlotDeprecated, pl.hivePlot`
  - *fate plots* `pl.fate`
  - *heatmap plots* `pl.causality, pl.comb_logic, pl.plot_hill_function, pl.response`
  - *Predictions plots* `pl.lap_min_time`
  - *External functionality* `ext.normalize_layers_pearson_residuals,`
    `ext.select_genes_by_pearson_residuals, ext.sctransform`

- More differential geometry analyses

  - include the `switch` mode in rank_jacobian_genes
  - added calculation of `sensitivity` matrix and relevant ranking 

- most probable path and *in silico* perturbation prediction

  - implemented least action path optimization (can be done in high dimensional space) with analytical Jacobian 
  - include genetic perturbation prediction by either changing the vector field function or simulate genetic perturbation via analytical Jacobian

- preprocessor class implementation

  - extensible modular preprocess steps 
  - support following recipes: monocle (dynamo), seurat (seurat V3 flavor), sctransform (seurat), pearson residuals and pearson residuals for feature selection, combined with monocle recipe (ensure no negative values)
  -  following recipes tested on zebrafish dataset to make implemetation results consistent:
    - monocle, seurat, pearson residuals
- CDlib integration

  - leiden, louvain, infomap community detection for cell clustering 
  - wrappers in `dyn.tl.*` for computing clusters
  - wrappers in `dyn.pl.*` for plotting

### Tutorial Updates on Readthedocs

- human HSC hematopoiesis RNA velocity analysis tutorials
- *in silico* perturbation and least action path (LAP) predictions tutorials on HSC dataset
- differential geometry analysis on HSC dataset

  - Molecular mechanism of megakaryocytes
  - Minimal network for basophil lineage commitment
  - Cell-wise analyses: dominant interactions
- gallery: Pancreatic endocrinogenesis differential geometry

Sample Dataset Updates

### CI/CD Updates

- update dynamo testing and pytest structure
- test building workflow on 3.7, 3.8, 3.9 (3.6 no longer tested on github building CI)

Performance Improvements

### API Changes

- preprocess

 - `pp.pca` -> `pca.pca_monocle`
- Native implementation of various graphical calculus using Numpy without using igraph. 

### Other Changes

- **general code refactor and bug fixing**
- **pl.scatters** refactor