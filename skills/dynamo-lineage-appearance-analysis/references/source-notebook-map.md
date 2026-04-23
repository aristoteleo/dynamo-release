# Source Notebook Map

Notebook inspected:

- `docs/tutorials/notebooks/400_tutorial_hsc_dynamo_megakaryocytes_appearance.ipynb`

## Notebook To Skill Mapping

### Notebook Intro And Setup

Notebook cells:

- data load with `dyn.sample_data.hematopoiesis()`
- environment and plotting setup

Mapped skill resources:

- `SKILL.md` Goal
- `SKILL.md` Input Contract
- `references/worked-example.md`

Reusable decision:

- keep the data-contract details
- drop notebook-only `%autoreload` and font styling

### Topography And Fixed-Point Curation

Notebook cells:

- `dyn.vf.topography(...)`
- `dyn.pl.topography(...)`
- manual `Xss` and `ftype` subsetting

Mapped skill resources:

- `SKILL.md` Stage Selection
- `SKILL.md` Minimal Execution Patterns
- `references/stage-selection.md`
- `references/compatibility.md`

Reusable decision:

- preserve the pattern of remapping then manually curating fixed points
- demote the concrete fixed-point indices to a worked example

### Vector-Field Pseudotime / Potential Comparison

Notebook cells:

- `build_graph(...)`
- `div(...)`
- `potential(...)`
- `potential_fp` and `pseudotime_fp`
- ECDF visualizations

Mapped skill resources:

- `SKILL.md` Quick Workflow
- `SKILL.md` Minimal Execution Patterns
- `references/stage-selection.md`
- `references/source-grounding.md`

Reusable decision:

- preserve the graph-operator spine
- drop the notebook-specific seaborn styling

### Molecular Mechanism / Jacobian

Notebook cells:

- `Meg_genes = ['FLI1', 'KLF1']`
- `dyn.vf.jacobian(...)`
- `dyn.pl.jacobian(...)`
- expression plots for `FLI1` and `KLF1`

Mapped skill resources:

- `SKILL.md` Minimal Execution Patterns
- `SKILL.md` Validation
- `references/worked-example.md`
- `references/source-grounding.md`

Reusable decision:

- keep the regulator-pair workflow
- treat `FLI1` and `KLF1` as worked-example defaults

### Speed / Divergence / Acceleration / Curvature

Notebook cells:

- `dyn.vf.speed(...)`
- `dyn.vf.divergence(...)`
- `dyn.vf.acceleration(...)`
- `dyn.vf.curvature(...)`
- notebook-style visualization panel

Mapped skill resources:

- `SKILL.md` Minimal Execution Patterns
- `references/stage-selection.md`
- `references/source-grounding.md`
- `references/compatibility.md`

Reusable decision:

- keep the quantity-computation spine
- drop the exact notebook subplot layout

### Narrative Conclusion

Notebook cells:

- final biological interpretation and schematic image

Mapped skill resources:

- not promoted into the main workflow

Reusable decision:

- retain the analysis steps that support the conclusion
- do not encode the conclusion itself as if it were universally true for every dataset
