.. automodule:: dynamo

API
===

Import dynamo as::

   import dynamo as dyn

Preprocessing (pp)
~~~~~~~~~~~~~~~~~~

.. autosummary::
      :maxdepth: 2
      :toctree: .

   pp.recipe_monocle

Tools (tl)
~~~~~~~~~~

.. autosummary::
    :maxdepth: 2
    :toctree: .

   tl.moments
   tl.dynamics

   tl.cell_velocities
   tl.reduceDimension

   tl.gene_wise_confidence
   tl.cell_wise_confidence


Conventional scRNA-seq (csc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

Time-resolved metabolic labeling based scRNA-seq (tsc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .


Vector field (vf)
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   vf.reconstruction
   vf.topography
   vf.divergence
   vf.curl
   vf.speed
   vf.acceleration
   vf.torsion
   vf.Potential

Prediction (pd)
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   pd.fate

   pd.fate_bias


Simulation (sim)
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .


External (ext)
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .
    ext.ddhodge
    ext.scribe
    ext.scifate

Plotting (pl)
~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   pl.scatters
   pl.phase_portraits
   pl.dynamics
   pl.umap
   pl.tsne
   pl.cell_wise_vectors
   pl.grid_vectors
   pl.streamline_plot
   pl.line_integral_conv
   pl.divergence
   pl.curl
   pl.topography
   pl.jacobian
   pl.nneighbors
   pl.kinetic_curves
   pl.kinetic_heatmap


Moive (mv)
~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   mv.fates
