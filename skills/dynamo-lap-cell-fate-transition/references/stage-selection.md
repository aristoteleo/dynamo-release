# Stage Selection

## When to Start from Stage 1 (LAP Computation)

Always start from Stage 1 when `transition_graph` has not been computed yet or when the cell type list, vector field, or EM_steps need to change.

Skip to later stages when `transition_graph` and `cells_indices` are already persisted to disk via `save_pickle`.

## `marginal_method` Options

Controls which cells are selected to represent each cell type attractor.

| value | behavior |
|---|---|
| `combined` | integrates distance, degree, and potential — default; most robust |
| `distance` | selects cells closest in embedding space to the attractor |
| `degree` | selects cells by graph connectivity degree |
| `potential` | selects cells by ddhodge potential value (requires `potential_column` to be meaningful) |

Use `combined` unless you have reason to bias toward one selection criterion.

## `basis_list` and `adj_key` Options

- `basis_list=['umap', 'pca']` — default; computes LAPs in both spaces. UMAP is used for visualization, PCA for downstream TF analysis.
- Set `umap_adj_key='X_umap_distances'` — this key is created by `dyn.tl.neighbors(adata, basis='umap', result_prefix='umap')`.
- Set `pca_adj_key='cosine_transition_matrix'` — this key is created during RNA velocity estimation.

If only UMAP-basis visualization is needed and TF analysis is not required, `basis_list=['umap']` is a valid shortcut.

## `EM_steps` Parameter

- `EM_steps=2` is the tutorial default; higher values improve cell selection but increase runtime.
- For smoke paths or exploratory runs, `EM_steps=1` reduces computation significantly.

## `lap_method` Options in `extract_transition_metrics`

| value | behavior |
|---|---|
| `action` | returns only action functional values per transition |
| `action_t` | returns both action and LAP integration time (in hours for labeling data) |

Use `action_t` when reporting transition times, which is biologically meaningful for scNT-seq data because the RNA velocity has absolute time units (hours).

## `plot_type` / `transition_type` Options in TF Analysis

| value | behavior |
|---|---|
| `development` | differentiating paths from progenitor (e.g., HSC→Meg) |
| `reprogramming` | reverse paths back to progenitor state (e.g., Meg→HSC) |
| `transdifferentiation` | lateral transitions between mature cell types (e.g., Mon→Bas) |

Call `analyze_transition_tfs` or `plot_transition_tf_analysis` once per transition type to generate the corresponding figure panel.

## `transitions_config` Structure

The `standard` key lists transitions handled with a single TF set per transition. The `special` key handles transitions with two alternative known TF sets:

```python
TRANSITIONS_CONFIG = {
    "standard": [
        "HSC->Meg", "HSC->Ery", "HSC->Bas", "HSC->Mon", "HSC->Neu",
        "Meg->HSC", "Meg->Neu", "Ery->Mon", "Mon->Meg", "Mon->Ery",
        "Mon->Bas", "Neu->Bas"
    ],
    "special": {
        "Ery->Neu": {
            "sets": [
                ("TFs1", "TFs_rank1", "Ery->Neu1"),
                ("TFs2", "TFs_rank2", "Ery->Neu2"),
            ]
        }
    }
}
```

Extend with additional `special` entries if other transitions have multiple competing TF literatures.

## Three-Step vs. All-in-One Pattern

Use `analyze_transition_tfs` (all-in-one) when:
- you want a single function call that produces processed rankings, the matrix, and a plot for one `plot_type`
- you do not need to inspect or modify intermediate results

Use the three-step pattern (`process_all_transition_rankings → create_reprogramming_matrix → plot_transition_tf_analysis`) when:
- you want to inspect `processed_rankings` before building the matrix
- you want different `total_tf_count` or `known_tfs_dict` at each step
- you want to plot multiple `transition_type` panels from the same `reprogramming_df`

## ROC Analysis Scope

`analyze_tf_roc_performance` uses all available transitions by default. Pass `transitions_to_include` to evaluate only a subset:

```python
development_transitions = ["HSC->Meg", "HSC->Ery", "HSC->Bas", "HSC->Mon", "HSC->Neu"]
roc_results = dyn.pd.analyze_tf_roc_performance(
    processed_rankings=processed_rankings,
    transitions_to_include=development_transitions,
)
```

The `roc_plot_params` dict is forwarded to `plot_roc_curve`. Supported keys: `figsize`, `fontsize`, `linewidth`, `roc_color`, `diagonal_color`, `title`, `legend_size`.
