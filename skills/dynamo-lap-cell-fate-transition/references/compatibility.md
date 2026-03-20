# Compatibility Notes

## API Changes in dynamo 1.4.2+

- `compute_cell_type_transitions` was introduced in dynamo 1.4.2 as a high-level wrapper that replaces manual cell-index selection and individual `dyn.pd.least_action` calls. Earlier versions required iterating over cell type pairs manually. Do not use the old per-pair `least_action` pattern unless the installed version predates 1.4.2.

## `dyn.pd.plot_kinetic_heatmap` vs `dyn.pl.kinetic_heatmap`

Two functions with similar names exist:

- `dyn.pd.plot_kinetic_heatmap(adata, cells_indices_dict, source_cell_type, target_cell_type, ...)` — prediction module; takes cell type names and a cells_indices dict; computes and plots kinetics along a LAP.
- `dyn.pl.kinetic_heatmap(adata, genes, mode=..., basis=..., tkey=..., ...)` — plot module; takes a gene list and plot mode; used for ddhodge pseudotime and vector-field heatmaps.

These are not interchangeable. The LAP tutorial uses `dyn.pd.plot_kinetic_heatmap`.

## `extract_transition_metrics` `lap_method` Default

- Default is `lap_method='action'` in the function signature.
- The tutorial uses `lap_method='action_t'` to also extract transition time (hours).
- If only action is needed, omit the argument; if time is needed, set `lap_method='action_t'` explicitly.

## `create_reprogramming_matrix` Null Safety Bug

`create_reprogramming_matrix` calls `transition_pmids.get(...)` and `transition_types.get(...)` without a None guard, despite the signature declaring `transition_pmids=None` and `transition_types=None` as defaults. This raises `AttributeError: 'NoneType' object has no attribute 'get'` when either argument is omitted.

**Workaround**: always pass empty dicts explicitly:

```python
reprogramming_dict, reprogramming_df = dyn.pd.create_reprogramming_matrix(
    transition_graph=transition_graph,
    transitions_config=TRANSITIONS_CONFIG,
    transition_pmids={},    # must be {} not None
    transition_types={},    # must be {} not None
    total_tf_count=133,
)
```

## `cloudpickle` Dependency for Pickle

`dyn.utils.save_pickle` falls back to `cloudpickle` when standard `pickle` fails (e.g., for lambda functions or complex objects embedded in the GeneTrajectory). Ensure `cloudpickle` is installed in the target environment if `save_pickle` raises a fallback warning.

## `umap_ddhodge_potential` Availability

The `umap_ddhodge_potential` column is written by `dyn.ext.ddhodge(adata, basis='umap', ...)`. It is pre-populated in `dyn.sample_data.hematopoiesis()` but will be missing on a freshly computed conventional dataset. Compute it with:

```python
dyn.ext.ddhodge(adata, basis='umap', sampling_method='velocity')
```

before calling `compute_cell_type_transitions`.

## `cosine_transition_matrix` Availability

`adata.obsp['cosine_transition_matrix']` is written during `dyn.tl.cell_velocities(adata, basis='pca', ...)`. It is present in the pre-processed hematopoiesis sample but may be absent on a freshly built dataset. Ensure it exists before calling `plot_kinetic_heatmap` or `analyze_kinetic_genes` with `adj_key='cosine_transition_matrix'`.

## `enable_gene_analysis` and `gtraj` Availability

When `enable_gene_analysis=False` is passed to `compute_cell_type_transitions`, the `gtraj` GeneTrajectory object and `ranking` DataFrame are NOT added to `transition_graph[name]`. This means:

- `extract_transition_metrics` will log `Error processing <transition>: 'gtraj'` for each pair but still returns the action and time DataFrames correctly.
- `process_all_transition_rankings` will fail because it reads `transition_graph[name]['ranking']`.

**Rule**: Use `enable_gene_analysis=True` (the default) whenever TF prioritization, kinetic heatmaps, or ROC analysis are needed. Use `enable_gene_analysis=False` only when you want action/time matrices but no TF analysis.

## `enable_plotting=False` for Headless Runs

Set `enable_plotting=False` in `compute_cell_type_transitions` when running in a headless or non-interactive environment to suppress matplotlib display calls that may block execution.

## `EM_steps` and Runtime

`EM_steps=2` is the tutorial default. For datasets with many cell types or large cell counts, each EM step substantially increases runtime. On the hematopoiesis dataset with 6 cell types and EM_steps=2, the full `compute_cell_type_transitions` call can take tens of minutes. Reduce to `EM_steps=1` for exploratory runs.
