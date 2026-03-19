# Source Grounding

This skill was generated from live code inspection of the dynamo source and the 501_lap_tutorial.ipynb notebook.

## Primary Source Files

Notebook inspected:

- `docs/tutorials/notebooks/501_lap_tutorial.ipynb`

Code inspected:

- `dynamo/prediction/least_action_path.py`
- `dynamo/prediction/_tf_eval.py`
- `dynamo/tools/connectivity.py`
- `dynamo/tools/utils.py`
- `dynamo/utils.py`

## Inspected Interfaces

### `dyn.pd.compute_cell_type_transitions`

Source: `dynamo/prediction/least_action_path.py`

Observed signature:

```python
def compute_cell_type_transitions(
    adata,
    cell_types,
    potential_column='umap_ddhodge_potential',
    cell_type_column='cell_type',
    reference_cell_types=None,
    basis_list=['umap', 'pca'],
    umap_adj_key='X_umap_distances',
    pca_adj_key='cosine_transition_matrix',
    EM_steps=2,
    top_genes=5,
    enable_plotting=True,
    enable_gene_analysis=True,
    marginal_method='combined',
    verify_selection=False,
    manual_cell_indices=None,
    manual_source_indices=None,
    manual_target_indices=None
)
```

Observed `marginal_method` branches:

- `combined` — default; integrates distance, degree, and potential
- `distance` — selects by spatial distance to attractor
- `degree` — selects by graph degree in neighbor graph
- `potential` — selects by ddhodge potential value

Returns: `(transition_graph, cells_indices_dict)`

Storage in `transition_graph[name]`:

- `LAP_umap`: dict with `prediction` (list of arrays) and `action` (list of floats)
- `LAP_pca`: dict with `prediction` and `action`
- `lap_results`: dict with basis-specific LAP objects
- `ranking`: DataFrame with gene MSD rankings (column `all`)
- `gtraj`: GeneTrajectory object
- `top_genes`: list of top dynamic gene names

### `dyn.pd.extract_transition_metrics`

Source: `dynamo/prediction/least_action_path.py`

Observed signature:

```python
def extract_transition_metrics(
    transition_graph,
    cells_indices_dict,
    cell_types,
    transcription_factors,
    top_tf_genes=10,
    lap_method='action'
)
```

Observed `lap_method` branches:

- `action` — extract only action values
- `action_t` — extract both action and transition time (used in the hematopoiesis tutorial)

Returns: `(action_df, t_df, tf_genes_results)`

- `action_df`: N×N DataFrame of action values (NaN for missing transitions)
- `t_df`: N×N DataFrame of LAP integration times
- `tf_genes_results`: dict keyed by transition name

### `dyn.pd.plot_kinetic_heatmap`

Source: `dynamo/prediction/least_action_path.py`

Observed signature:

```python
def plot_kinetic_heatmap(
    adata,
    cells_indices_dict,
    source_cell_type,
    target_cell_type,
    transcription_factors,
    basis='pca',
    adj_key='cosine_transition_matrix',
    figsize=(16, 8),
    color_map='bwr',
    font_scale=0.8,
    scaler=0.6,
    save_path=None,
    show_plot=True,
    return_data=False
)
```

Note: This is `dyn.pd.plot_kinetic_heatmap` (prediction module), distinct from `dyn.pl.kinetic_heatmap` (plot module), which has a completely different signature.

### `dyn.pd.analyze_kinetic_genes`

Source: `dynamo/prediction/least_action_path.py`

Observed signature:

```python
def analyze_kinetic_genes(
    adata,
    cells_indices_dict,
    source_cell_type,
    target_cell_type,
    transcription_factors,
    top_genes=20,
    basis='pca',
    adj_key='cosine_transition_matrix'
)
```

Returns: `(ranking_df, top_tfs_list)`

- `ranking_df`: DataFrame with MSD scores and TF flag columns
- Internally calls `dyn.tl.rank_genes(adata, 'traj_msd')`

### `dyn.pd.analyze_transition_tfs`

Source: `dynamo/prediction/_tf_eval.py`

Observed signature:

```python
def analyze_transition_tfs(
    transition_graph,
    human_tfs_names,
    transitions_config,
    plot_type='transdifferentiation',
    known_tfs_dict=None,
    transition_pmids=None,
    transition_types=None,
    total_tf_count=133,
    transition_color_dict=None,
    figsize=(8, 5)
)
```

Observed `plot_type` branches:

- `development`
- `reprogramming`
- `transdifferentiation`

Returns: `(processed_rankings, reprogramming_dict, reprogramming_df)`

### `dyn.pd.process_all_transition_rankings`

Source: `dynamo/prediction/_tf_eval.py`

Observed signature:

```python
def process_all_transition_rankings(
    transition_graph,
    human_tfs_names,
    known_tfs_dict=None
)
```

Writes into `transition_graph[name]`:

- `TFs`: boolean array marking which genes are TFs
- `TFs_rank`: rank position array
- `TFs1`, `TFs2`, `TFs_rank1`, `TFs_rank2` for `Ery->Neu` special case

Returns: `processed_rankings` dict

### `dyn.pd.create_reprogramming_matrix`

Source: `dynamo/prediction/_tf_eval.py`

Observed signature:

```python
def create_reprogramming_matrix(
    transition_graph,
    transitions_config,
    transition_pmids=None,
    transition_types=None,
    total_tf_count=133
)
```

`transitions_config` structure:

```python
{
    "standard": ["HSC->Meg", ...],   # processed with TFs/TFs_rank
    "special": {
        "Ery->Neu": {
            "sets": [("TFs1", "TFs_rank1", "Ery->Neu1"), ("TFs2", "TFs_rank2", "Ery->Neu2")]
        }
    }
}
```

`reprogramming_df` columns: `genes`, `rank` (normalized 0–1 as priority score), `transition`, `type`

### `dyn.pd.plot_transition_tf_analysis`

Source: `dynamo/prediction/_tf_eval.py`

Observed signature:

```python
def plot_transition_tf_analysis(
    reprogramming_df,
    transition_type='transdifferentiation',
    figsize=(8, 5),
    score_threshold=0.8,
    transition_color_dict=None
)
```

Default color dict: `{"development": "#2E3192", "reprogramming": "#EC2227", "transdifferentiation": "#B9519E"}`

Returns: `(fig, ax)` tuple

### `dyn.pd.analyze_tf_roc_performance`

Source: `dynamo/prediction/_tf_eval.py`

Observed signature:

```python
def analyze_tf_roc_performance(
    processed_rankings,
    transitions_to_include=None,
    plot_roc=True,
    roc_plot_params=None
)
```

Returns dict keys: `consolidated_df`, `fpr`, `tpr`, `roc_auc`, `performance_summary`

### `dyn.pd.consolidate_processed_rankings`

Source: `dynamo/prediction/_tf_eval.py`

Observed signature:

```python
def consolidate_processed_rankings(
    processed_rankings,
    transitions_to_include=None
)
```

Adds `source` column for transition of origin; returns concatenated DataFrame.

### `dyn.pd.calculate_priority_scores_from_consolidated`

Source: `dynamo/prediction/_tf_eval.py`

Observed signature:

```python
def calculate_priority_scores_from_consolidated(consolidated_df)
```

Priority score formula: `1 - (rank_position / reference_size)`. Adds `priority_score` column.

### `dyn.pd.plot_roc_curve`

Source: `dynamo/prediction/_tf_eval.py`

Observed signature:

```python
def plot_roc_curve(
    y_true,
    y_scores,
    figsize=(4, 4),
    fontsize=12,
    linewidth=1.5,
    roc_color='darkorange',
    diagonal_color='navy',
    title=None,
    legend_size=12,
    hide_zero_ticks=True,
    return_fig=False
)
```

Returns: `(fpr, tpr, roc_auc)` by default, or matplotlib axes if `return_fig=True`.

### `dyn.pd.get_tf_statistics`

Source: `dynamo/prediction/_tf_eval.py`

Observed signature:

```python
def get_tf_statistics(processed_rankings, reprogramming_df)
```

Returns dict: `n_all_tfs`, `n_valid_tfs`, `n_overlap`, `overlap_percentage`, `all_tfs`, `valid_tfs`, `overlap_tfs`

### `dyn.tl.neighbors`

Source: `dynamo/tools/connectivity.py`

Observed signature:

```python
def neighbors(
    adata,
    X_data=None,
    genes=None,
    basis='pca',
    layer=None,
    n_pca_components=30,
    n_neighbors=30,
    method=None,
    metric='euclidean',
    metric_kwads=None,
    cores=1,
    seed=19491001,
    result_prefix='',
    **kwargs
)
```

With `basis='umap', result_prefix='umap'`, writes:

- `adata.obsp['umap_connectivities']`
- `adata.obsp['umap_distances']`
- `adata.uns['umap_neighbors']`

The LAP tutorial calls `neighbors(adata, basis='umap', result_prefix='umap')` to create `X_umap_distances` required by `compute_cell_type_transitions`.

Observed `method` branches when autoselected:

- `kd_tree` (default for small/moderate datasets)
- `ball_tree`
- `brute`
- `umap` (UMAP graph)

### `dyn.tl.select_cell`

Source: `dynamo/tools/utils.py`

Observed signature:

```python
def select_cell(
    adata,
    grp_keys,
    grps,
    presel=None,
    mode='union',
    output_format='index'
)
```

Observed branches:

- `mode`: `union`, `intersection`
- `output_format`: `index`, `mask`

Returns: numpy array of cell indices or boolean mask.

### `dyn.utils.save_pickle` / `dyn.utils.load_pickle`

Source: `dynamo/utils.py`

Observed signatures:

```python
def save_pickle(file, path)
def load_pickle(path)
```

Falls back to `cloudpickle` if standard `pickle` fails. Used to persist `transition_graph` (complex object graph with LAP metadata and GeneTrajectory objects).
