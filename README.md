[![package](https://github.com/aristoteleo/dynamo-release/workflows/Python%20package/badge.svg)](https://github.com/aristoteleo/dynamo-release/runs/950435412) 
[![package](https://github.com/aristoteleo/dynamo-release/workflows/Upload%20Python%20Package/badge.svg)](https://pypi.org/project/dynamo-release/) 
[![documentation](https://readthedocs.org/projects/dynamo-release/badge/?version=latest)](https://dynamo-release.readthedocs.io/en/latest/)
![build](https://github.com/aristoteleo/dynamo-release/actions/workflows/python-package.yml/badge.svg)
![test](https://github.com/aristoteleo/dynamo-release/actions/workflows/python-plain-run-test.yml/badge.svg)

## **Dynamo**: Mapping Vector Field of Single Cells

Inclusive model of expression dynamics with metabolic labeling based scRNA-seq / multiomics, vector field reconstruction, potential landscape mapping and differential geometry analyses.

[Installation](https://dynamo-release.readthedocs.io/en/latest/ten_minutes_to_dynamo.html#how-to-install) - [Ten minutes to dynamo](https://dynamo-release.readthedocs.io/en/latest/ten_minutes_to_dynamo.html) - [Tutorials](https://dynamo-release.readthedocs.io/en/latest/notebooks/Differential_geometry.html) - [API](https://dynamo-release.readthedocs.io/en/latest/API.html) - [Citation](https://www.biorxiv.org/content/10.1101/696724v2) - [Theory](https://dynamo-release.readthedocs.io/en/latest/notebooks/Primer.html)

![Dynamo](https://user-images.githubusercontent.com/7456281/93838270-11d8da00-fc57-11ea-94de-d11b529731e1.png)


Single-cell RNA-seq, together with RNA velocity and metabolic labeling, reveals cellular states and transitions at unprecedented resolution. Fully exploiting these data, however, requires dynamical models capable of predicting cell fate and unveiling the governing regulatory mechanisms. Here, we introduce dynamo, an analytical framework that reconciles intrinsic splicing and labeling kinetics to estimate absolute RNA velocities, reconstructs velocity vector fields that predict future cell fates, and finally employs differential geometry analyses to elucidate the underlying regulatory networks. We applied dynamo to a wide range of disparate biological processes including prediction of future states of differentiating hematopoietic stem cell lineages, deconvolution of glucocorticoid responses from orthogonal cell-cycle progression, characterization of regulatory networks driving zebrafish pigmentation, and identification of possible routes of resistance to SARS-CoV-2 infection. Our work thus represents an important step in going from qualitative, metaphorical conceptualizations of differentiation, as exemplified by Waddington’s epigenetic landscape, to quantitative and predictive theories.

## Highlights of dynamo
* Robust and accurate estimation of RNA velocities for regular scRNA-seq datasets:
    * Three methods for the velocity estimations (including the new negative binomial distribution based approach)
    * Improved kernels for transition matrix calculation and velocity projection 
    * Strategies to correct RNA velocity vectors (when your RNA velocity direction is problematic) 
* Inclusive modeling of time-resolved metabolic labeling based scRNA-seq:
    * Explicitly model RNA metabolic labeling, in conjunction with RNA bursting, transcription, splicing and degradation
    * Comprehensive RNA kinetic rate estimation for one-shot, pulse, chase and mixture metabolic labeling experiments
* Move beyond RNA velocity to continuous vector field function for functional and predictive analyses of cell fate transitions:
    * Dynamical systems approaches to identify stable cell types (fixed points), boundaries of cell states (separatrices), etc
    * Calculate RNA acceleration (reveals early drivers), curvature (reveals master regulators of fate decision points), divergence (stability of cell states) and RNA Jacobian (cell-state dependent regulatory networks) 
    * Various downstream differential geometry analyses to rank critical regulators/effectors,  and visualize regulatory networks at key fate decision points    

## Discussion 
Please use github issue tracker to report coding related [issues](https://github.com/aristoteleo/dynamo-release/issues) of dynamo. For community discussion of novel usage cases, analysis tips and biological interpretations of dynamo, please join our public slack workspace: [dynamo-discussion](https://join.slack.com/t/dynamo-discussionhq/shared_invite/zt-itnzjdxs-PV~C3Hr9uOArHZcmv622Kg) (Only a working email address is required from the slack side).

## Contribution 
If you want to contribute to the development of dynamo, please check out CONTRIBUTION instruction: [Contribution](https://github.com/aristoteleo/dynamo-release/blob/master/CONTRIBUTING.md)
