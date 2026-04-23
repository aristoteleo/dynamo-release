# Source Notebook Map

Source notebook:
`docs/tutorials/notebooks/403_Differential_geometry.ipynb`

Use this file to see how the notebook was converted into a reusable downstream differential-geometry skill.

Conversion rule used here:

- the stable skill identity is downstream `dynamo` differential-geometry analysis on a conventional RNA-velocity vector field
- zebrafish pigmentation is the worked example, not the trigger surface

## Notebook Sections To Skill Responsibilities

### 1. Load Or Preprocess Data

Notebook role:

- load `dyn.sample_data.zebrafish()`
- preprocess and fit dynamics

Preserved in the skill:

- `SKILL.md` Quick Workflow
- `SKILL.md` Input Contract
- `references/stage-selection.md`
- `references/source-grounding.md`

Source-grounded addition:

- the current runtime provides raw zebrafish counts, so the reusable skill treats preprocessing as a real bootstrap stage instead of assuming a processed object

### 2. Learn The PCA-Space Vector Field

Notebook role:

- compute PCA-space velocity
- fit `VectorField(..., basis='pca')`

Preserved in the skill:

- `SKILL.md` Interface Summary
- `SKILL.md` Minimal Execution Patterns
- `references/stage-selection.md`

Source-grounded addition:

- reviewer execution found that smaller subsets may need `transition_genes=adata.var.use_for_pca.values` for the `basis='pca'` velocity branch

### 3. Velocity, Acceleration, And Curvature Ranking

Notebook role:

- rank velocity genes
- compute acceleration and curvature
- rank acceleration and curvature genes

Preserved in the skill:

- `SKILL.md` Stage Selection
- `SKILL.md` Minimal Execution Patterns
- `SKILL.md` Validation

Source-grounded addition:

- the skill records the real `method` and `formula` branches, not only the notebook's default path

### 4. Gene Set Enrichment

Notebook role:

- run `dyn.ext.enrichr(...)` on selected ranked genes

Conversion choice:

- not part of the core skill identity
- intentionally left out of the main workflow because it is a downstream interpretation consumer of ranking outputs, not a defining step of differential-geometry analysis

### 5. Jacobian Calculation, Ranking, And Network Construction

Notebook role:

- mark top PCA genes
- compute Jacobian
- rank Jacobian-derived quantities
- build and plot cluster-specific networks

Preserved in the skill:

- `SKILL.md` Minimal Execution Patterns
- `references/jacobian-and-network-analysis.md`

Source-grounded additions:

- explicit `sampling` branches on `jacobian(...)`
- explicit `mode` branches on `rank_jacobian_genes(...)`
- explicit returned edge-list shape from `build_network_per_cluster(...)`

### 6. ddhodge Pseudotime And Heatmaps

Notebook role:

- run `ddhodge(...)`
- plot kinetic heatmaps for expression, velocity, acceleration, and curvature

Preserved in the skill:

- `SKILL.md` Stage Selection
- `references/pseudotime-and-state-graph.md`

Source-grounded additions:

- explicit `adjmethod`, `sampling_method`, `mode`, and `gene_order_method` branches
- explicit storage key `adata.uns['kinetics_heatmap']`

### 7. State Graph

Notebook role:

- run `state_graph(..., method='vf')`
- plot the resulting state graph

Preserved in the skill:

- `SKILL.md` Minimal Execution Patterns
- `references/pseudotime-and-state-graph.md`

Source-grounded addition:

- the skill treats `method='markov'` and `method='naive'` as real alternate branches even though the notebook demonstrates only `vf`

### 8. Export And Cleanup

Notebook role:

- export rank tables to Excel
- call `cleanup(...)`

Preserved in the skill:

- `SKILL.md` Minimal Execution Patterns
- `SKILL.md` Constraints
- `references/source-grounding.md`

Source-grounded addition:

- `cleanup(...)` removes `kinetics_heatmap`, so the skill keeps export and cleanup at the very end

## What Was Intentionally Not Carried Over

- zebrafish-specific biological interpretation prose
- figure-by-figure plot commentary
- direct presentation choices such as plotting style or notebook display layout
- enrichment analysis as a default branch

## When To Reopen The Notebook

Reopen the notebook only when:

- the user wants zebrafish-specific figure parity
- the user wants the same exact genes or cell types used in the teaching narrative
- a future notebook revision changes the demonstrated order of vector-field, Jacobian, or ddhodge stages
