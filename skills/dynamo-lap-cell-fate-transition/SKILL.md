---
name: dynamo-lap-cell-fate-transition
description: Compute least action paths (LAP) between hematopoietic or general cell types in a dynamo vector-field AnnData, then rank transcription factors by MSD along each path and evaluate predictions via ROC analysis. Use when running dyn.pd.compute_cell_type_transitions, predicting optimal cell fate conversion trajectories, prioritizing transcription factor cocktails for cell reprogramming, or reproducing the 501_lap_tutorial.ipynb workflow on scNT-seq or metabolic-labeling data.
---

# Dynamo Least Action Path Cell Fate Transition

## Goal

Given a `dynamo` AnnData with a completed vector field and ddhodge potential in UMAP space, compute pairwise least action paths (LAPs) between all requested cell types, extract per-path action and time metrics, rank transcription factors by their mean square displacement (MSD) along each path, and evaluate TF prioritization quality via ROC analysis.

## Quick Workflow

1. Confirm the input AnnData has `VecFld_umap`, `umap_ddhodge_potential`, and a cell type column.
2. Build a UMAP-basis neighbor graph with `dyn.tl.neighbors(..., basis='umap', result_prefix='umap')`.
3. Run `dyn.pd.compute_cell_type_transitions(...)` to compute LAPs for all requested cell type pairs.
4. Persist `transition_graph` and `cells_indices` to disk with `dyn.utils.save_pickle(...)` — this step is long-running.
5. Load a human TF list with `dyn.sample_data.human_tfs()`.
6. Extract action and time matrices with `dyn.pd.extract_transition_metrics(...)`.
7. Plot kinetic heatmaps of gene expression dynamics along selected paths with `dyn.pd.plot_kinetic_heatmap(...)`.
8. Rank TFs across all transitions with `dyn.pd.analyze_transition_tfs(...)`.
9. Evaluate TF ranking quality with `dyn.pd.analyze_tf_roc_performance(...)`.

## Interface Summary

- `dyn.tl.neighbors(adata, X_data=None, genes=None, basis='pca', layer=None, n_pca_components=30, n_neighbors=30, method=None, metric='euclidean', metric_kwads=None, cores=1, seed=19491001, result_prefix='', **kwargs)` — build neighbor graph; use `basis='umap', result_prefix='umap'` to create the `X_umap_distances` adjacency needed by LAP.
- `dyn.pd.compute_cell_type_transitions(adata, cell_types, potential_column='umap_ddhodge_potential', cell_type_column='cell_type', reference_cell_types=None, basis_list=['umap', 'pca'], umap_adj_key='X_umap_distances', pca_adj_key='cosine_transition_matrix', EM_steps=2, top_genes=5, enable_plotting=True, enable_gene_analysis=True, marginal_method='combined', verify_selection=False, manual_cell_indices=None, manual_source_indices=None, manual_target_indices=None)` — core LAP computation; returns `(transition_graph, cells_indices)`.
- `dyn.pd.extract_transition_metrics(transition_graph, cells_indices_dict, cell_types, transcription_factors, top_tf_genes=10, lap_method='action')` — extract action and time DataFrames per transition pair; returns `(action_df, t_df, tf_genes)`.
- `dyn.pd.plot_kinetic_heatmap(adata, cells_indices_dict, source_cell_type, target_cell_type, transcription_factors, basis='pca', adj_key='cosine_transition_matrix', figsize=(16, 8), color_map='bwr', font_scale=0.8, scaler=0.6, save_path=None, show_plot=True, return_data=False)` — plot gene expression dynamics along a LAP.
- `dyn.pd.analyze_kinetic_genes(adata, cells_indices_dict, source_cell_type, target_cell_type, transcription_factors, top_genes=20, basis='pca', adj_key='cosine_transition_matrix')` — rank TFs by MSD along one selected LAP; returns `(ranking_df, top_tfs)`.
- `dyn.pd.analyze_transition_tfs(transition_graph, human_tfs_names, transitions_config, plot_type='transdifferentiation', known_tfs_dict=None, transition_pmids=None, transition_types=None, total_tf_count=133, transition_color_dict=None, figsize=(8, 5))` — all-in-one: process rankings, build matrix, and plot for one transition type.
- `dyn.pd.process_all_transition_rankings(transition_graph, human_tfs_names, known_tfs_dict=None)` — step-wise version: add TF and known_TF columns to all transitions; returns `processed_rankings`.
- `dyn.pd.create_reprogramming_matrix(transition_graph, transitions_config, transition_pmids=None, transition_types=None, total_tf_count=133)` — build normalized priority-score DataFrame; returns `(reprogramming_dict, reprogramming_df)`. **Important**: pass `{}` not `None` for `transition_pmids` and `transition_types` — the current source does not guard against None (see `references/compatibility.md`).
- `dyn.pd.plot_transition_tf_analysis(reprogramming_df, transition_type='transdifferentiation', figsize=(8, 5), score_threshold=0.8, transition_color_dict=None)` — bar/scatter plot of known TF priority scores per transition type.
- `dyn.pd.analyze_tf_roc_performance(processed_rankings, transitions_to_include=None, plot_roc=True, roc_plot_params=None)` — compute and optionally plot ROC curve for TF ranking; returns dict with `fpr`, `tpr`, `roc_auc`, `consolidated_df`.
- `dyn.pd.get_tf_statistics(processed_rankings, reprogramming_df)` — return counts and overlap of TFs; dict with `n_all_tfs`, `n_valid_tfs`, `n_overlap`, `overlap_percentage`.
- `dyn.utils.save_pickle(file, path)` / `dyn.utils.load_pickle(path)` — persist and restore any Python object; falls back to `cloudpickle` if standard pickle fails.

Read `references/source-grounding.md` for inspected signatures and storage key evidence.

## Stage Selection

- **Stage 1 — LAP computation**: `compute_cell_type_transitions` with `enable_plotting=True, enable_gene_analysis=True`. This is the slow step. Save outputs with `save_pickle` before proceeding.
- **Stage 2 — Metrics and visualization**: `extract_transition_metrics`, LAP path plotting, action/time heatmaps. Load from pickle if Stage 1 was already run.
- **Stage 3 — Kinetic heatmaps**: `plot_kinetic_heatmap` and `analyze_kinetic_genes` for specific source–target pairs. Requires `cells_indices` and the adata with fitted vector field.
- **Stage 4 — TF ranking and ROC**: `analyze_transition_tfs` or the three-step pattern (`process_all_transition_rankings → create_reprogramming_matrix → plot_transition_tf_analysis`), then `analyze_tf_roc_performance`.

Use `analyze_transition_tfs` (all-in-one) for the quickest path. Use the three-step pattern when you need fine-grained control over known TF dictionaries, PMID tables, or plot parameters.

Read `references/stage-selection.md` for details on `marginal_method`, `lap_method`, and `plot_type` branch options.

## Input Contract

- `AnnData` with `VecFld_umap` in `.uns` (vector field must already be fitted).
- `adata.obs[potential_column]` exists (default `umap_ddhodge_potential`); populated by `dyn.ext.ddhodge`.
- `adata.obsm['X_umap']` and `adata.obsm['X_pca']` exist.
- `adata.obsp['cosine_transition_matrix']` exists before calling PCA-basis LAP or kinetic heatmap.
- A cell type label column in `adata.obs` (default `cell_type`).
- A transcription factor name list (e.g., from `dyn.sample_data.human_tfs()` for human datasets).
- The hematopoiesis worked example loads from `dyn.sample_data.hematopoiesis()` — it already includes all required fields.

## Minimal Execution Patterns

### Stage 1 — LAP computation

```python
import dynamo as dyn

adata = dyn.sample_data.hematopoiesis()

# Build UMAP neighbor graph needed for LAP
dyn.tl.neighbors(adata, basis="umap", result_prefix="umap")

cell_types = ["HSC", "Meg", "Ery", "Bas", "Mon", "Neu"]

transition_graph, cells_indices = dyn.pd.compute_cell_type_transitions(
    adata=adata,
    cell_types=cell_types,
    reference_cell_types=["HSC"],
    marginal_method="combined",
    potential_column="umap_ddhodge_potential",
    cell_type_column="cell_type",
    EM_steps=2,
    top_genes=5,
    enable_plotting=True,
    enable_gene_analysis=True,
)

# Persist — this step is expensive
dyn.utils.save_pickle(transition_graph, "result/transition_graph.pkl")
dyn.utils.save_pickle(cells_indices, "result/cells_indices.pkl")
adata.write("result/adata_labeling_analysis.h5ad")
```

### Stage 2 — Metrics extraction

```python
transition_graph = dyn.utils.load_pickle("result/transition_graph.pkl")
cells_indices = dyn.utils.load_pickle("result/cells_indices.pkl")

human_tfs = dyn.sample_data.human_tfs()
human_tfs_names = list(human_tfs["Symbol"])

action_df, t_df, tf_genes = dyn.pd.extract_transition_metrics(
    transition_graph=transition_graph,
    cells_indices_dict=cells_indices,
    cell_types=cell_types,
    transcription_factors=human_tfs_names,
    top_tf_genes=10,
    lap_method="action_t",   # 'action' or 'action_t'
)
```

### Stage 3 — Kinetic heatmap for one pair

```python
ranking, top_tfs = dyn.pd.analyze_kinetic_genes(
    adata=adata,
    cells_indices_dict=cells_indices,
    source_cell_type="HSC",
    target_cell_type="Bas",
    transcription_factors=human_tfs_names,
    top_genes=15,
)

dyn.pd.plot_kinetic_heatmap(
    adata=adata,
    cells_indices_dict=cells_indices,
    source_cell_type="HSC",
    target_cell_type="Bas",
    transcription_factors=human_tfs_names,
    save_path="figures/HSC_to_Bas.png",
    figsize=(16, 4),
)
```

### Stage 4 — TF ranking and ROC

```python
KNOWN_TFS_DICT = {
    "HSC->Meg": ["GATA1", "GATA2", "ZFPM1", "GFI1B", "FLI1", "NFE2"],
    # ... add other transitions
}
TRANSITIONS_CONFIG = {
    "standard": [
        "HSC->Meg", "HSC->Ery", "HSC->Bas", "HSC->Mon", "HSC->Neu",
        "Meg->HSC", "Meg->Neu", "Ery->Mon", "Mon->Meg", "Mon->Ery",
        "Mon->Bas", "Neu->Bas"
    ],
    "special": {"Ery->Neu": {"sets": [("TFs1", "TFs_rank1", "Ery->Neu1"), ("TFs2", "TFs_rank2", "Ery->Neu2")]}},
}

# All-in-one path
processed_rankings, reprogramming_dict, reprogramming_df = dyn.pd.analyze_transition_tfs(
    transition_graph=transition_graph,
    human_tfs_names=human_tfs_names,
    transitions_config=TRANSITIONS_CONFIG,
    plot_type="transdifferentiation",
    known_tfs_dict=KNOWN_TFS_DICT,
    total_tf_count=133,
)

# ROC analysis
roc_results = dyn.pd.analyze_tf_roc_performance(
    processed_rankings=processed_rankings,
    plot_roc=True,
    roc_plot_params={"figsize": (3, 3), "fontsize": 12, "legend_size": 12},
)
print("AUC:", roc_results["roc_auc"])
```

## Validation

After LAP computation, confirm:

- `transition_graph` contains keys for each `A->B` pair
- Each entry has `LAP_umap` with `prediction` and `action` sub-keys
- `cells_indices` contains one entry per cell type in `cell_types`
- Pickle files written successfully to disk

After metrics extraction, confirm:

- `action_df.shape == (len(cell_types), len(cell_types))`
- `t_df.shape == (len(cell_types), len(cell_types))`
- `tf_genes` is a dict keyed by transition names

After kinetic heatmap, confirm:

- Output figure file exists at `save_path` if provided
- `ranking` DataFrame contains MSD scores
- `top_tfs` list is non-empty

After TF ranking and ROC, confirm:

- `processed_rankings` contains one entry per transition
- Each entry has `TFs` and `TFs_rank` (or `TFs1`/`TFs2` for special cases)
- `reprogramming_df` columns include `genes`, `rank`, `transition`, `type`
- `roc_results['roc_auc']` is a float between 0 and 1

## Constraints

- `compute_cell_type_transitions` requires `VecFld_umap` and `umap_ddhodge_potential` to be precomputed.
- Run `dyn.tl.neighbors(adata, basis='umap', result_prefix='umap')` before `compute_cell_type_transitions`; it creates `adata.obsp['X_umap_distances']` required for UMAP-basis LAP.
- The `EM_steps` parameter controls the number of iterative optimization rounds; lower values (1–2) are faster for smoke paths; default `2` is used in the tutorial.
- Do not treat the hematopoiesis cell type labels, KNOWN_TFS_DICT, or TRANSITION_PMIDS dicts as required skill inputs — they are hematopoiesis-specific worked example parameters.
- The `transitions_config['special']` key handles transitions with two alternative known TF sets (e.g., `Ery->Neu`). Define it explicitly if your dataset has such ambiguity.
- `dyn.pd.plot_kinetic_heatmap` is a prediction module function distinct from `dyn.pl.kinetic_heatmap`; they have different signatures and purposes.
- Set `enable_gene_analysis=True` (default) whenever TF prioritization or kinetic heatmaps are needed; with `enable_gene_analysis=False`, `gtraj` and `ranking` are absent from `transition_graph`, causing `process_all_transition_rankings` to fail and `extract_transition_metrics` to skip TF extraction with a logged error.

## Resource Map

- Read `references/stage-selection.md` when choosing `marginal_method`, `lap_method`, or `plot_type`.
- Read `references/source-grounding.md` for full inspected signatures and storage key evidence.
- Read `references/source-notebook-map.md` to trace `501_lap_tutorial.ipynb` sections to skill resources.
- Read `references/compatibility.md` when encountering API name mismatches or missing attributes.
- Use `assets/acceptance.json` for the bounded smoke paths used by local acceptance.
