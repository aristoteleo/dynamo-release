# Source Notebook Map

Notebook: `docs/tutorials/notebooks/501_lap_tutorial.ipynb`

## Section → Skill Resource Mapping

| Notebook section | Skill resource |
|---|---|
| Introduction: LAP method and hematopoiesis background | `SKILL.md` Goal; biology demoted to worked example |
| Import and data load (`dyn.sample_data.hematopoiesis()`) | `SKILL.md` Minimal Execution Patterns — Stage 1 |
| Compute neighbor graph (`dyn.tl.neighbors`) | `SKILL.md` Interface Summary; `references/source-grounding.md` |
| Run pairwise LAP (`compute_cell_type_transitions`) | `SKILL.md` Stage 1 pattern; `references/stage-selection.md` |
| Cell attractor scatter plot | Display-only; demoted from skill body |
| Save/load pickle (`save_pickle`, `load_pickle`) | `SKILL.md` Stage 1 constraint note |
| Visualize LAP paths on UMAP streamline | Visualization code; demoted from skill body |
| LAP time bar chart (developmental lineages) | `SKILL.md` Stage 2 pattern; worked example only |
| Action and time heatmaps | `SKILL.md` Stage 2 validation checks |
| Kinetic heatmap (`plot_kinetic_heatmap`) | `SKILL.md` Stage 3 pattern; `references/source-grounding.md` |
| `analyze_kinetic_genes` TF ranking | `SKILL.md` Stage 3 pattern |
| `KNOWN_TFS_DICT`, `TRANSITIONS_CONFIG` | `SKILL.md` Stage 4 pattern; note: hematopoiesis-specific |
| `analyze_transition_tfs` (all-in-one) | `SKILL.md` Stage 4 pattern; `references/stage-selection.md` |
| Three-step pattern (process → matrix → plot) | `SKILL.md` Stage 4 pattern; `references/stage-selection.md` |
| ROC curve analysis (`analyze_tf_roc_performance`) | `SKILL.md` Stage 4 pattern; `references/source-grounding.md` |
| Custom ROC plot (`plot_roc_curve`) | `references/source-grounding.md` |
| `get_tf_statistics` | `references/source-grounding.md` |

## Key Worked-Example Values (Hematopoiesis-Specific)

The following values appear as tutorial defaults. They are hematopoiesis-specific and should not be treated as skill requirements:

- `cell_types = ["HSC", "Meg", "Ery", "Bas", "Mon", "Neu"]`
- `potential_column = "umap_ddhodge_potential"`
- `cell_type_column = "cell_type"`
- `KNOWN_TFS_DICT` — literature-curated TF lists per hematopoietic transition
- `TRANSITION_PMIDS` — PubMed IDs specific to hematopoiesis literature
- `TRANSITION_TYPES` — classification as development / reprogramming / transdifferentiation
- `total_tf_count = 133` — number of detectable human TFs in this dataset
- `human_tfs_names` — from `dyn.sample_data.human_tfs()`

## Notebook Cells Not Included in Skill Body

- Inline matplotlib cell scatter plots for attractor visualization
- Commented-out load cells for restarting a session
- `%matplotlib inline` magic commands
- Direct `dyn.configuration.set_pub_style(...)` style-setting calls
- Inline LAP path visualization (`plot_lap` helper function)
- Print statements for intermediate transition names

## AnnData State at Notebook Load Time

The hematopoiesis sample dataset (`dyn.sample_data.hematopoiesis()`) loads in a fully preprocessed state:

- `obs`: includes `cell_type`, `umap_ddhodge_potential`, `umap_ddhodge_div`, and many computed columns
- `uns`: includes `VecFld_pca`, `VecFld_umap`, `neighbors`, `grid_velocity_umap`, etc.
- `obsm`: includes `X_pca`, `X_umap`, `velocity_pca`, `velocity_umap`, etc.
- `obsp`: includes `cosine_transition_matrix`, `connectivities`, `distances`, etc.

The `neighbors(adata, basis='umap', result_prefix='umap')` call adds `umap_connectivities` and `umap_distances` to `obsp`, which are required by the UMAP-basis LAP computation.
