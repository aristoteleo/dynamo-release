|package| |PyPI| |Docs|

**Dynamo**: Mapping Vector Field of Single Cells
===================================================

.. image:: https://user-images.githubusercontent.com/7456281/93838270-11d8da00-fc57-11ea-94de-d11b529731e1.png
   :align: center

Single-cell RNA-seq, together with RNA velocity and metabolic labeling, reveals cellular states and transitions at unprecedented resolution. Fully exploiting these data, however, requires dynamical models capable of predicting cell fate and unveiling the governing regulatory mechanisms. Here, we introduce dynamo, an analytical framework that reconciles intrinsic splicing and labeling kinetics to estimate absolute RNA velocities, reconstructs velocity vector fields that predict future cell fates, and finally employs differential geometry analyses to elucidate the underlying regulatory networks. We applied dynamo to a wide range of disparate biological processes including prediction of future states of differentiating hematopoietic stem cell lineages, deconvolution of glucocorticoid responses from orthogonal cell-cycle progression, characterization of regulatory networks driving zebrafish pigmentation, and identification of possible routes of resistance to SARS-CoV-2 infection. Our work thus represents an important step in going from qualitative, metaphorical conceptualizations of differentiation, as exemplified by Waddington’s epigenetic landscape, to quantitative and predictive theories.

.. |Docs| image:: https://readthedocs.org/projects/dynamo-release/badge/?version=latest
   :target: https://dynamo-release.readthedocs.io

.. |package| image:: https://github.com/aristoteleo/dynamo-release/workflows/Python%20package/badge.svg
   :target: https://github.com/aristoteleo/dynamo-release/runs/950435412

.. |PyPI| image:: https://github.com/aristoteleo/dynamo-release/workflows/Upload%20Python%20Package/badge.svg
   :target: https://pypi.org/project/dynamo-release/


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

   Introduction
   Primer


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

   zebrafish


.. toctree::
   :caption: Labeling scRNA-seq
   :maxdepth: 1
   :hidden:

   scNT_seq_readthedocs
   scEU_seq_rpe1_analysis_kinetic
   scEU_seq_organoid_analysis_kinetic


.. toctree::
   :caption: Differential geometry
   :maxdepth: 1
   :hidden:

   Differential_geometry


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _`dynamo`: https://github.com/aristoteleo/dynamo-release
.. _`issues`: (https://github.com/aristoteleo/dynamo-release/issues)
.. _`dynamo-discussion`: https://join.slack.com/t/dynamo-discussionhq/shared_invite/zt-ghve9pzp-r9oJ9hSQznWrDcx1fCog6g
.. _`Contribution`: https://github.com/aristoteleo/dynamo-release/blob/master/CONTRIBUTING.md

