|package| |PyPI| |Docs|

**Dynamo**: Mapping Vector Field of Single Cells
===================================================

.. image:: https://user-images.githubusercontent.com/7456281/152110270-7ee1b0ed-1205-495d-9d65-59c7984d2fa2.png
   :align: center

Single-cell (sc)RNA-seq, together with RNA velocity and metabolic labeling, reveals cellular states and transitions at unprecedented resolution. Fully exploiting these data, however, requires kinetic models capable of unveiling governing regulatory functions. Here, we introduce an analytical framework dynamo, which infers absolute RNA velocity, reconstructs continuous vector fields that predict cell fates, employs differential geometry to extract underlying regulations, and ultimately predicts optimal reprogramming paths and perturbation outcomes. We highlight dynamo’s power to overcome fundamental limitations of conventional splicing-based RNA velocity analyses to enable accurate velocity estimations on a metabolically labeled human hematopoiesis scRNA-seq dataset. Furthermore, differential geometry analyses reveal mechanisms driving early megakaryocyte appearance and elucidate asymmetrical regulation within the PU.1-GATA1 circuit. Leveraging the least-action-path method, dynamo accurately predicts drivers of numerous hematopoietic transitions. Finally, in silico perturbations predict cell-fate diversions induced by gene perturbations. Dynamo, thus, represents an important step in advancing quantitative and predictive theories of cell-state transitions.

.. |Docs| image:: https://readthedocs.org/projects/dynamo-release/badge/?version=latest
   :target: https://dynamo-release.readthedocs.io

.. |package| image:: https://github.com/aristoteleo/dynamo-release/workflows/Python%20package/badge.svg
   :target: https://github.com/aristoteleo/dynamo-release/runs/950435412

.. |PyPI| image:: https://github.com/aristoteleo/dynamo-release/workflows/Upload%20Python%20Package/badge.svg
   :target: https://pypi.org/project/dynamo-release/
   

Highlights of dynamo
====================
#. Robust and accurate estimation of RNA velocities for regular scRNA-seq datasets:
    * Three methods for the velocity estimations (including the new negative binomial distribution based approach)
    * Improved kernels for transition matrix calculation and velocity projection
    * Strategies to correct RNA velocity vectors (when your RNA velocity direction is problematic)
#. Inclusive modeling of time-resolved metabolic labeling based scRNA-seq:
    * Explicitly model RNA metabolic labeling, in conjunction with RNA bursting, transcription, splicing and degradation
    * Comprehensive RNA kinetic rate estimation for one-shot, pulse, chase and mixture metabolic labeling experiments
#. Move beyond RNA velocity to continuous vector field function for functional and predictive analyses of cell fate transitions:
    * Dynamical systems approaches to identify stable cell types (fixed points), boundaries of cell states (separatrices), etc
    * Calculate RNA acceleration (reveals early drivers), curvature (reveals master regulators of fate decision points), divergence (stability of cell states) and RNA Jacobian (cell-state dependent regulatory networks)
    * Various downstream differential geometry analyses to rank critical regulators/effectors,  and visualize regulatory networks at key fate decision points
#. Non-trivial vector field prediction of cell fate transitions:
    * Least action path approach to predict the optimal paths and transcriptomic factors of cell fate reprogrammings
    * In silico perturbation to predict the gene-wise perturbation effects and cell fate diversion after genetic perturbations


News
==========
.. _Cell: https://www.sciencedirect.com/science/article/pii/S0092867421015774#tbl1

#. After 3.5+ years of perseverance, our dynamo paper :cite:p:`QIU2022` is finally online in Cell_!

Discussion
==========
Please use github issue tracker to report coding related `issues`_ of dynamo. For community discussion of novel usage cases, analysis tips and biological interpretations of dynamo, please join our public slack workspace: `dynamo-discussion`_ (Only a working email address is required from the slack side).

Contribution
============
If you want to contribute to the development of dynamo, please check out CONTRIBUTION instruction: `Contribution`_


.. toctree::
   :caption: Introduction
   :maxdepth: 1
   :hidden:

   notebooks/Introduction
   notebooks/Primer
   notebooks/lap_box_introduction
   notebooks/perturbation_introduction_theory.rst



.. toctree::
   :caption: Contents
   :maxdepth: 3

   ten_minutes_to_dynamo
   API
   Class
   FAQ
   Release_notes
   Reference
   Acknowledgement


.. toctree::
   :caption: Conventional scRNA-seq
   :maxdepth: 1
   :hidden:

   notebooks/zebrafish


.. toctree::
   :caption: Labeling scRNA-seq
   :maxdepth: 1
   :hidden:

   notebooks/scNT_seq_readthedocs
   notebooks/scEU_seq_rpe1_analysis_kinetic
   notebooks/scEU_seq_organoid_analysis_kinetic


.. toctree::
   :caption: Differential geometry
   :maxdepth: 1
   :hidden:

   notebooks/Differential_geometry.rst

.. toctree::
   :caption: Vector field predictions
   :maxdepth: 1
   :hidden:

   notebooks/lap_tutorial/lap_tutorial
   notebooks/perturbation_tutorial/perturbation_tutorial

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _`dynamo`: https://github.com/aristoteleo/dynamo-release
.. _`issues`: (https://github.com/aristoteleo/dynamo-release/issues)
.. _`dynamo-discussion`: https://join.slack.com/t/dynamo-discussionhq/shared_invite/zt-ghve9pzp-r9oJ9hSQznWrDcx1fCog6g
.. _`Contribution`: https://github.com/aristoteleo/dynamo-release/blob/master/CONTRIBUTING.md

