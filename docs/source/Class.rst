Class
=====

Estimation
----------

Conventional scRNA-seq (est.csc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dynamo.est.csc.ss_estimation
    :members:
    :inherited-members:

.. autoclass:: dynamo.est.csc.velocity
    :members:
    :inherited-members:


Time-resolved metabolic labeling based scRNA-seq (est.tsc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Base class: a general estimation framework*

.. autoclass:: dynamo.est.tsc.kinetic_estimation
    :members:
    :inherited-members:

*Deterministic models via analytical solution of ODEs*

.. autoclass:: dynamo.est.tsc.Estimation_DeterministicDeg
    :members:
    :inherited-members:

.. autoclass:: dynamo.est.tsc.Estimation_DeterministicDegNosp
    :members:
    :inherited-members:

.. autoclass:: dynamo.est.tsc.Estimation_DeterministicKinNosp
    :members:
    :inherited-members:

.. autoclass:: dynamo.est.tsc.Estimation_DeterministicKin
    :members:
    :inherited-members:

*Stochastic models via matrix form of moment equations*

.. autoclass:: dynamo.est.tsc.Estimation_MomentDeg
    :members:
    :inherited-members:

.. autoclass:: dynamo.est.tsc.Estimation_MomentDegNosp
    :members:
    :inherited-members:

.. autoclass:: dynamo.est.tsc.Estimation_MomentKin
    :members:
    :inherited-members:

.. autoclass:: dynamo.est.tsc.Estimation_MomentKinNosp
    :members:
    :inherited-members:

*Mixture models for kinetic / degradation experiments*

.. autoclass:: dynamo.est.tsc.Lambda_NoSwitching
    :members:
    :inherited-members:

.. autoclass:: dynamo.est.tsc.Mixture_KinDeg_NoSwitching
    :members:
    :inherited-members:

Vector field
------------

Vector field class
~~~~~~~~~~~~~~~~~~

.. autoclass:: dynamo.vf.vectorfield
    :members:
    :inherited-members:

.. autoclass:: dynamo.vf.Pot
    :members:
    :inherited-members:


Movie
-----

Animation class
~~~~~~~~~~~~~~~

.. autoclass:: dynamo.mv.StreamFuncAnim
    :members:
    :inherited-members:
