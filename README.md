<p align="center">
  <img height="150" src="https://dynamo-release.readthedocs.io/en/latest/_static/logo_with_word.png" />
</p>

##

[![package](https://github.com/aristoteleo/dynamo-release/workflows/Python%20package/badge.svg)](https://github.com/aristoteleo/dynamo-release/runs/950435412) 
[![package](https://github.com/aristoteleo/dynamo-release/workflows/Upload%20Python%20Package/badge.svg)](https://pypi.org/project/dynamo-release/) 
[![documentation](https://readthedocs.org/projects/dynamo-release/badge/?version=latest)](https://dynamo-release.readthedocs.io/en/latest/)
![build](https://github.com/aristoteleo/dynamo-release/actions/workflows/python-package.yml/badge.svg)
![test](https://github.com/aristoteleo/dynamo-release/actions/workflows/python-plain-run-test.yml/badge.svg)


## **Dynamo**: Mapping Transcriptomic Vector Fields of Single Cells

Inclusive model of expression dynamics with metabolic labeling based scRNA-seq / multiomics, vector field reconstruction, potential landscape mapping, differential geometry analyses, and most probably paths / *in silico* perturbation predictions.

[Installation](https://dynamo-release.readthedocs.io/en/latest/ten_minutes_to_dynamo.html#how-to-install) - [Ten minutes to dynamo](https://dynamo-release.readthedocs.io/en/latest/ten_minutes_to_dynamo.html) - [Tutorials](https://dynamo-release.readthedocs.io/en/latest/notebooks/Differential_geometry.html) - [API](https://dynamo-release.readthedocs.io/en/latest/API.html) - [Citation](https://www.biorxiv.org/content/10.1101/696724v2) - [Theory](https://dynamo-release.readthedocs.io/en/latest/notebooks/Primer.html)

![Dynamo](https://user-images.githubusercontent.com/7456281/152110270-7ee1b0ed-1205-495d-9d65-59c7984d2fa2.png)

Single-cell (sc)RNA-seq, together with RNA velocity and metabolic labeling, reveals cellular states and transitions at unprecedented resolution. Fully exploiting these data, however, requires kinetic models capable of unveiling governing regulatory functions. Here, we introduce an analytical framework dynamo, which infers absolute RNA velocity, reconstructs continuous vector fields that predict cell fates, employs differential geometry to extract underlying regulations, and ultimately predicts optimal reprogramming paths and perturbation outcomes. We highlight dynamo’s power to overcome fundamental limitations of conventional splicing-based RNA velocity analyses to enable accurate velocity estimations on a metabolically labeled human hematopoiesis scRNA-seq dataset. Furthermore, differential geometry analyses reveal mechanisms driving early megakaryocyte appearance and elucidate asymmetrical regulation within the PU.1-GATA1 circuit. Leveraging the least-action-path method, dynamo accurately predicts drivers of numerous hematopoietic transitions. Finally, in silico perturbations predict cell-fate diversions induced by gene perturbations. Dynamo, thus, represents an important step in advancing quantitative and predictive theories of cell-state transitions.

## Highlights of dynamo
* Robust and accurate estimation of RNA velocities for regular scRNA-seq datasets:
    * Three methods for the velocity estimations (including the new negative binomial distribution based approach)
    * Improved kernels for transition matrix calculation and velocity projection 
    * Strategies to correct RNA velocity vectors (when your RNA velocity direction is problematic) 
* Inclusive modeling of time-resolved metabolic labeling based scRNA-seq:
    * Overcome intrinsic limitation of the conventional splicing based RNA velocity analyses
    * Explicitly model RNA metabolic labeling, in conjunction with RNA bursting, transcription, splicing and degradation
    * Comprehensive RNA kinetic rate estimation for one-shot, pulse, chase and mixture metabolic labeling experiments
* Move beyond RNA velocity to continuous vector field function for gaining mechannistic insights of cell fate transitions:
    * Dynamical systems approaches to identify stable cell types (fixed points), boundaries of cell states (separatrices), etc
    * Calculate RNA acceleration (reveals early drivers), curvature (reveals master regulators of fate decision points), divergence (stability of cell states) and RNA Jacobian (cell-state dependent regulatory networks) 
    * Various downstream differential geometry analyses to rank critical regulators/effectors,  and visualize regulatory networks at key fate decision points    
* Non-trivial vector field predictions of cell fate transitions:
    * Least action path approach to predict the optimal paths and transcriptomic factors of cell fate reprogrammings
    * In silico perturbation to predict the gene-wise perturbation effects and cell fate diversion after genetic perturbations

## News
* 2/1/2022: after 3.5+ years of perseverance, our dynamo paper is finally online in [Cell](https://www.sciencedirect.com/science/article/pii/S0092867421015774#tbl1) today!
* 2/15/2022: primers and tutorials on least action paths and in silico perturbation are released.
* 3/14/2022: Since today dynamo has its own logo! Here the arrow represents the RNA velocity vector field, while the helix the RNA molecule and the colored dots RNA metabolic labels (4sU labeling). See [readthedocs](https://dynamo-release.readthedocs.io/en/latest/index.html)
* 4/4/2022: dynamo 1.1 released

## Discussion 
Please use github issue tracker to report coding related [issues](https://github.com/aristoteleo/dynamo-release/issues) of dynamo. For community discussion of novel usage cases, analysis tips and biological interpretations of dynamo, please join our public slack workspace: [dynamo-discussion](https://join.slack.com/t/dynamo-discussionhq/shared_invite/zt-itnzjdxs-PV~C3Hr9uOArHZcmv622Kg) (Only a working email address is required from the slack side). 

## Contribution 
If you want to contribute to the development of dynamo, please check out CONTRIBUTION instruction: [Contribution](https://github.com/aristoteleo/dynamo-release/blob/master/CONTRIBUTING.md)
