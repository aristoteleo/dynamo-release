# Branch Selection

This notebook mixes sample-data assumptions, curated-gene preprocessing, time-group smoothing, one-shot kinetic fitting, custom total-RNA velocity calculation, and visualization. Use this reference to keep those stages separate instead of replaying every cell literally.

## Default Job

If the user asks to reproduce or adapt one-shot total RNA velocity in `dynamo`:

1. inspect whether the input already has `new`, `total`, `time`, and an optional curated gene list
2. preprocess with the monocle path
3. run `moments(..., group='time')`
4. force `adata.uns["pp"]["has_splicing"] = False`
5. run `dynamics(..., one_shot_method='sci_fate', model='deterministic')`
6. run `calculate_velocity_alpha_minus_gamma_s(...)`
7. project with `cell_velocities(..., X=M_t, V=velocity_alpha_minus_gamma_s, method='cosine')`

## Stage 1: Gene List And Preprocessing

Relevant branch-heavy interfaces:

- `Preprocessor.preprocess_adata(..., recipe='monocle')`
- `Preprocessor.preprocess_adata_monocle(...)`
- `Preprocessor.config_monocle_recipe(..., n_top_genes=...)`

Observed `recipe` branches in current source:

- `monocle`
- `seurat`
- `sctransform`
- `pearson_residuals`
- `monocle_pearson_residuals`

Recommendation for this workflow family:

- use `recipe='monocle'`
- use `force_gene_list` plus `config_monocle_recipe(..., n_top_genes=len(gene_list))` when a curated list exists

When to switch:

- drop `force_gene_list` only when the user does not have a trusted notebook-derived or marker/HVG list
- use another `recipe` only when the user explicitly wants that preprocessing family and accepts that the exact notebook path is no longer being reproduced

Important runtime rule:

- `anndata.concat(...)` can drop `uns["genes_to_use"]`
- preserve the list explicitly before rebuilding subsets

## Stage 2: Grouped Moments

Relevant branch-heavy parameters on `dyn.tl.moments(...)`:

- `group`
- `use_gaussian_kernel`
- `use_mnn`
- `layers`

Recommendation for this workflow family:

- use `group='time'`
- keep `use_gaussian_kernel=False`
- keep `use_mnn=False`
- keep `layers='all'`

Reason:

- the one-shot fit depends on smoothed `M_n`, `M_t`, and `M_s`
- empirical notebook-adjacent execution populated those layers and `obsp["moments_con"]` through the default grouped branch

When to switch:

- omit `group` only when the user explicitly wants global smoothing without time-specific neighborhoods
- use `use_mnn=True` only for batch-correction-aware neighbor logic, not as a default notebook conversion choice

## Stage 3: Model-2 Kinetics

Relevant branch-heavy parameters on `dyn.tl.dynamics(...)` for this workflow family:

- `group`
- `model`
- `est_method`
- `one_shot_method`

Observed `one_shot_method` branches in current source:

- `combined`
- `sci-fate`
- `sci_fate`

Observed `model` branches in current source include:

- `deterministic`
- `stochastic`
- `mixture`
- `model_selection`

Recommendation for this workflow family:

- set `adata.uns["pp"]["has_splicing"] = False`
- use `group='time'`
- use `one_shot_method='sci_fate'`
- use `model='deterministic'`

Why:

- this reproduces the tutorial's Model-2 path
- empirically checked `sci_fate` and `combined` branches both succeeded on the hematopoiesis sample
- the default deterministic one-shot runs stored `velocity_T` and `est_method='ols'`

When to switch:

- use `one_shot_method='combined'` when the user explicitly wants the steady-state-plus-absolute-gamma branch
- do not silently switch to a splicing-aware model when the goal is total RNA velocity from the tutorial

Important storage rule:

- current runtime does not store the chosen `one_shot_method` in `adata.uns["dynamics"]`
- document the branch choice in your own workflow notes if later interpretation depends on it

## Stage 4: Total RNA Velocity Layer

Relevant interface:

- `calculate_velocity_alpha_minus_gamma_s(adata, gene_subset_key='use_for_pca', velocity_layer_name='velocity_alpha_minus_gamma_s')`

Recommendation:

- keep the default `gene_subset_key='use_for_pca'`
- keep the default output layer name unless the user explicitly wants a separate comparison run

Important runtime rules:

- the function requires `adata.obs["time"]`
- it reads `adata.layers["M_n"]` and `adata.layers["M_s"]`
- it expects per-time kinetic parameters already written by `dynamics(...)`

## Stage 5: Low-Dimensional Projection

Relevant branch-heavy parameters on `dyn.tl.cell_velocities(...)`:

- `method`: `kmc`, `fp`, `cosine`, `pearson`, `transform`
- `basis`
- `adj_key`

Recommendation for this workflow family:

- use `basis='umap'`
- use `X=adata.layers["M_t"]`
- use `V=adata.layers["velocity_alpha_minus_gamma_s"]`
- use `method='cosine'` for notebook parity

When to switch:

- use `method='pearson'` when the user wants the alternate correlation-style kernel
- use `method='transform'` only when `adata.uns["umap_fit"]` already exists
- use `kmc` or `fp` only when the user explicitly wants those Markov or operator branches and understands the extra tuning surface

Important storage rules:

- `cosine` writes `adata.obsp["cosine_transition_matrix"]`
- `pearson` writes `adata.obsp["pearson_transition_matrix"]`
- both empirically checked branches populated `adata.obsm["velocity_umap"]`

## Stage 6: Visualization

Relevant branch-heavy plotting parameters:

- `streamline_plot(..., method='gaussian' | 'SparseVFC')`
- `phase_portraits(..., show_quiver=False | True, use_smoothed=True | False, ...)`

Recommendation:

- use `streamline_plot(..., method='gaussian')` after `velocity_umap` exists
- treat phase portraits as an optional interpretation branch, not part of the execution spine

## Decision Rule

If the user asks for:

- "the notebook result" -> run the default job and keep the curated-gene-list branch if available
- "just the kinetic fit" -> stop after `dynamics(...)` and report that no low-dimensional total-RNA velocity has been projected yet
- "total RNA velocity on UMAP" -> continue through `calculate_velocity_alpha_minus_gamma_s(...)` and `cell_velocities(...)`
- "why projection changed" -> compare `method='cosine'` and `method='pearson'`
- "why my subset fails" -> first check whether `genes_to_use` was lost and whether each time group has enough cells
