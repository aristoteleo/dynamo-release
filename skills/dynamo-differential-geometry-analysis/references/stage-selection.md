# Stage Selection

This notebook mixes upstream velocity fitting, downstream differential geometry, ranking, network extraction, pseudotime, and reporting. Keep those jobs separate instead of replaying the notebook line by line.

## Capability Partition

- Core executable job:
  bootstrap a conventional spliced / unspliced `AnnData` into a PCA-space vector field, then compute differential-geometry quantities from that field
- Optional downstream analysis jobs:
  grouped ranking, Jacobian ranking, regulatory-network extraction, ddhodge pseudotime, kinetic heatmaps, and state graphs
- Optional visualization and reporting jobs:
  UMAP overlays, Jacobian plots, arc plots, circos plots, streamline plots, and Excel export
- Notebook-only pedagogy:
  zebrafish biology explanation, figure narration, and cell-by-cell interpretation prose

## Default Job

If the user asks for the reusable equivalent of `403_Differential_geometry.ipynb`:

1. preprocess with `recipe='monocle'`
2. run `dynamics(...)`
3. run `reduceDimension(...)`
4. compute velocity in the embedding and in PCA space
5. fit `VectorField(..., basis='pca', method='SparseVFC')`
6. branch into geometry ranking, Jacobian analysis, or ddhodge/state-graph analysis as needed

## Stage 1: Bootstrap From Raw Conventional Data

Relevant branch-heavy parameters:

- `Preprocessor.preprocess_adata(..., recipe=...)`
- `dyn.tl.reduceDimension(..., reduction_method=...)`
- `dyn.tl.cell_velocities(..., basis=None | 'pca')`

Observed `recipe` branches:

- `monocle`
- `seurat`
- `sctransform`
- `pearson_residuals`
- `monocle_pearson_residuals`

Observed `reduction_method` branches:

- `trimap`
- `diffusion_map`
- `tsne`
- `umap`
- `psl`
- `sude`

Recommendation:

- use `recipe='monocle'`
- use `reduction_method='umap'`
- compute both the default embedding velocity and `basis='pca'` velocity

Important runtime rule:

- on smaller subsets, prefer `dyn.tl.cell_velocities(..., basis='pca', transition_genes=adata.var.use_for_pca.values)` instead of relying on the narrower default call

## Stage 2: Vector Field And Scalar / Vector Geometry

Relevant branch-heavy parameters:

- `VectorField(..., method='SparseVFC' | 'dynode', pot_curl_div=False | True)`
- `acceleration(..., method='analytical' | 'numerical')`
- `curvature(..., formula=1 | 2, method='analytical' | 'numerical')`

Recommendation:

- use `basis='pca'`
- use `method='SparseVFC'`
- keep `pot_curl_div=False` unless the user explicitly wants curl or divergence on a 2D basis
- use `acceleration(..., method='analytical')`
- use `curvature(..., formula=2, method='analytical')`

When to switch:

- use `method='dynode'` only when that backend is installed and explicitly requested
- use `formula=1` only when the user explicitly wants the alternate curvature definition
- use `method='numerical'` only for comparison, debugging, or branch audits

## Stage 3: Jacobian And Ranking

Relevant branch-heavy parameters:

- `jacobian(..., sampling=None | 'random' | 'velocity' | 'trn', method='analytical' | 'numerical')`
- `rank_jacobian_genes(..., mode='full reg' | 'full eff' | 'reg' | 'eff' | 'int' | 'switch')`

Recommendation:

- shrink the gene universe first with `top_pca_genes(...)`
- use `jacobian(..., basis='pca', sampling='trn', method='analytical')`
- use `mode='full_reg'` or `mode='full_eff'` when you need whole-group ranking tables
- use `mode='reg'` or `mode='eff'` when you need target-specific summaries

When to switch:

- use `sampling='random'` or `sampling='velocity'` only when you explicitly want a different cell subset policy
- use `mode='int'` when you want pairwise interaction ranking
- use `mode='switch'` only when the task is to identify mutual inhibition candidates

## Stage 4: ddhodge Pseudotime, Heatmaps, And State Graphs

Relevant branch-heavy parameters:

- `ddhodge(..., adjmethod='graphize_vecfld' | 'naive', sampling_method='random' | 'velocity' | 'trn')`
- `kinetic_heatmap(..., mode='vector_field' | 'lap' | 'pseudotime', gene_order_method='maximum' | 'half_max_ordering' | 'raw')`
- `state_graph(..., method='vf' | 'markov' | 'naive')`

Recommendation:

- use `ddhodge(..., basis='pca', adjmethod='graphize_vecfld', sampling_method='velocity')`
- use `kinetic_heatmap(..., mode='pseudotime', tkey='pca_ddhodge_potential', gene_order_method='maximum')`
- use `state_graph(..., group='<group>', basis='pca', method='vf')`

When to switch:

- use `adjmethod='naive'` only if a relevant adjacency or transition matrix already exists and you explicitly want that branch
- use `mode='vector_field'` when the user wants ordering along the vector field without ddhodge pseudotime
- use `mode='lap'` or `method='markov'` only when the user specifically wants a Markov-style or graph-Laplacian interpretation

## Stage 5: Export And Cleanup

Relevant branch-heavy parameters:

- `export_rank_xlsx(..., rank_prefix='rank')`
- `cleanup(..., del_prediction=False | True, del_2nd_moments=False | True)`

Recommendation:

- export only after ranking keys are final
- run `cleanup(...)` only after plots or reports no longer depend on `kinetics_heatmap`, fate internals, or cached model fits

## Decision Rule

If the user asks for:

- "the notebook workflow" -> run the default job, but treat zebrafish labels and example genes as worked-example defaults
- "acceleration or curvature" -> stop after Stage 2
- "regulators or networks" -> complete Stages 1 to 3, then stop
- "pseudotime heatmap" or "state graph" -> complete Stages 1 to 4
- "export ranking tables" -> run the needed ranking stage first, then Stage 5
