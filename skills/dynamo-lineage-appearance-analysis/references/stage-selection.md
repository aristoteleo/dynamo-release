# Stage Selection

Use this reference to decide which subset of the notebook to reuse.

## 1. Fixed-Point And Topography Stage

Use this stage when the user needs to inspect candidate attractors, saddles, or manual fixed-point curation.

Default path:

- trust existing `adata.uns['VecFld_umap']` if it already contains plausible `Xss` and `ftype`
- run `dyn.vf.topography(adata, basis='umap', n=...)` only when fixed points need remapping
- plot with `dyn.pl.topography(..., basis='umap', fps_basis='umap')`

Branch notes:

- `n` controls how many samples are used for fixed-point search; manual fixed-point indices are not portable across different `n`
- `terms` defaults to `['streamline', 'fixed_points']`
- `fate='both'` maps to bidirectional integration when trajectories are drawn
- `quiver_source='raw'` is the safe default in the current runtime

Use this stage alone if the task is mostly landscape inspection.

## 2. Appearance-Order Comparison Stage

Use this stage when the user needs a defensible claim that one lineage appears earlier than another, not just a UMAP screenshot.

Core operators:

- `build_graph(adata.obsp['cosine_transition_matrix'])`
- `div(g)`
- `potential(g, -divergence)` for the cosine branch used in the notebook
- `build_graph(adata.obsp['fp_transition_rate'])`
- `potential(g_fp, fp_divergence)` for the notebook's Fokker-Planck branch

Interpretation notes:

- `potential(...)` returns a nonnegative shifted potential
- the notebook interprets smaller potential as earlier intrinsic time
- the notebook flips the sign for `pseudotime_fp = -potential_fp`

Use this stage whenever lineage-order claims are central to the answer.

## 3. Regulator-Pair Jacobian Stage

Use this stage when the user wants to test or visualize a specific regulatory interaction.

Default path:

- choose regulators and effectors explicitly
- run `dyn.vf.jacobian(..., basis='pca', method='analytical')`
- plot with `dyn.pl.jacobian(..., basis='umap', j_basis='pca')`

Branch notes:

- `sampling` can be `None`, `random`, `velocity`, or `trn`
- `method` can be `analytical` or `numerical`
- if `basis='umap'`, current source forces all cells instead of sampled subsets
- if `regulators` or `effectors` are strings matching a boolean column in `adata.var`, current code expands them to gene lists

Use this stage when the user asks about FLI1/KLF1-like gene-pair behavior or other lineage-defining regulators.

## 4. Vector-Calculus Stage

Use this stage when the user wants dynamic quantities rather than only a Jacobian map.

Default path:

- `dyn.vf.speed(adata, basis='pca', method='analytical')`
- `dyn.vf.divergence(adata, basis='pca', method='analytical')`
- `dyn.vf.acceleration(adata, basis='pca', method='analytical')`
- `dyn.vf.curvature(adata, basis='pca', formula=2, method='analytical')`

Branch notes:

- `speed(..., method='numeric')` falls back to stored velocity vectors instead of analytical field evaluation
- `divergence(...)` can reuse precomputed `jacobian_pca` entries when present
- `curvature(..., formula=2)` returns both the norm and matrix in the current design
- `curvature(..., formula=1)` is source-documented but failed in reviewer execution in this runtime

Use this stage when the user wants to compare how rapidly or sharply different lineages move through the field.
