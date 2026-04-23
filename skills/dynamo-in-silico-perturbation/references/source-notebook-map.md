# Source Notebook Map

Notebook: `docs/tutorials/notebooks/502_perturbation_tutorial.ipynb`

## Section → Skill Resource Mapping

| Notebook section | Skill resource |
|---|---|
| Introduction: perturbation method and paper context | `SKILL.md` Goal — demoted to capability framing |
| Import and data load (`dyn.sample_data.hematopoiesis()`) | `SKILL.md` Input Contract; worked example note |
| Define gene sets (murine_blood_cells, gran_lineage_genes, erythroid_differentiation) | Demoted — hematopoiesis-specific gene lists, not skill requirements |
| GATA1 suppression: `perturbation(adata, "GATA1", [-100], emb_basis="umap")` | `SKILL.md` Minimal Execution Patterns — single-gene suppression |
| SPI1 suppression: `perturbation(adata, "SPI1", [-100], emb_basis="umap")` | `SKILL.md` Minimal Execution Patterns — single-gene suppression (same pattern) |
| Joint SPI1+GATA1 suppression | `SKILL.md` Minimal Execution Patterns — multi-gene mixed |
| KLF1 activation: `perturbation(adata, "KLF1", [100], emb_basis="umap")` | `SKILL.md` Minimal Execution Patterns — single-gene activation |
| Triple GATA1+KLF1+TAL1 activation | `SKILL.md` Minimal Execution Patterns — multi-gene |
| `dyn.pl.streamline_plot(..., basis="umap_perturbation")` | `SKILL.md` Minimal Execution Patterns + Validation |
| Biological interpretation (which cells divert where) | Demoted — worked example pedagogy |

## Key Worked-Example Values (Hematopoiesis-Specific)

The following values appear as tutorial defaults. They are hematopoiesis-specific and should not be treated as skill requirements:

- `murine_blood_cells = ["RUN1T1", "HLF", "LMO2", "PRDM5", "PBX1", "ZFP37", "MYCN", "MEIS1"]`
- `gran_lineage_genes = ["CEBPE", "RUNX1T1", "KLF1", "CEBPA", "FOSB", "JUN", "SPI1", "ZC3HAV1"]`
- `erythroid_differentiation = ["GATA1", "TAL1", "LMO2", "KLF1", "MYB", "LDB1", "NFE2", "GFI1B", "BCL11A"]`
- Cell type labels (`"cell_type"` column) and their color scheme
- Expression saturation values (`-100`, `100`) chosen for hematopoiesis paper figures

## AnnData State at Notebook Load Time

The hematopoiesis sample dataset (`dyn.sample_data.hematopoiesis()`) loads in a fully preprocessed state including:

- `adata.uns['VecFld_pca']` — PCA-space vector field (required)
- `adata.uns['jacobian_pca']` — PCA-space Jacobian (required for default `j_delta_x` method)
- `adata.uns['PCs']`, `adata.uns['pca_mean']` — PCA projection matrices (required)
- `adata.obsm['X_pca']`, `adata.obsm['X_umap']` — embeddings (required)
- `adata.obsp['cosine_transition_matrix']` — transition matrix (used internally by `cell_velocities`)

## Capability Partition Decision

The notebook contains one tightly coupled job: perturbation + visualization. The perturbation output (`X_{emb_basis}_perturbation`) is directly consumed by the visualization call. Splitting into two skills would create a thin wrapper with no standalone value. A single skill with explicit stage notes is the correct design.

## Cells Not Included in Skill Body

- `%load_ext autoreload` and `%autoreload 2` — notebook-only
- `dyn.get_all_dependencies_version()` — diagnostic only
- `dyn.configuration.set_figure_params(...)`, `dyn.pl.style(...)` — style setup
- `figsize=(4,4)`, `dpi=80` inside `s_kwargs_dict` — notebook display preferences
- Inline gene set variable definitions that are hematopoiesis-specific
