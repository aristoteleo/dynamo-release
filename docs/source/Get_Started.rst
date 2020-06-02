Get Started
-----------

Welcome to dynamo!

Dynamo is a computational framework that includes an inclusive model of expression dynamics with scSLAM-seq and multiomics, vector field reconstruction and potential landscape mapping. 


How to install
^^^^^^^^^^^^^^
Please check the `Install Dynamo`_  for details on how to install dynamo. 


To process the scSLAM-seq data, please refer to the `NASC-seq analysis pipeline`_. We are also working on a command line tool for this too. For processing splicing data, you
can either use the `velocyto command line interface`_ or the `bustool from Pachter lab`_

Compatability
^^^^^^^^^^^^^
dynamo is fully compatabile with velocyto, scanpy and scvelo. So you can use your loom or annadata object as input for dynamo. The velocity vector samples estimated from either velocyto or scvelo can be also directly used to reconstruct the functional form of vector field 
and the potential landscape in the entire expression space. 


Basic Usage
^^^^^^^^^^^

Import dynamo as::

    import dynamo as dyn


Velocity vectors
''''''''''''''''
To analyze the multi-time-series or kinetics experiment, as described in our preprint_ (Fig 1B), we build an inclusive model of RNA bursting, processing, translation, degradation and metabolic labelling to study the gene expression dynamics. Dynamo efficiently estimates all kinetic parameters in this model, by taking advantage of the matrix form of the set of moment generating functions about different species in the model with a nonlinear least square fitting and Latin hypercube sampling method. Often we don't have the multi-time-series data, we then use the degenerate model introducted in the sections starting from Modeling the dynamic system using ODEs of our preprint_. To estimate parameters by the moment generating functions in dynamo::

    dyn.tl.moment(adata, **params)
    dyn.tl.estimation(adata, **params)

To use the velocity by in dynamo::

    dyn.tl.estimation(adata, **params)
    dyn.tl.estimation(adata, **params).vel_s(**params)
    dyn.tl.estimation(adata, **params).vel_u(**params)
    dyn.tl.estimation(adata, **params).vel_p(**params)

Vector field reconstruction
'''''''''''''''''''''''''''
In classical physics, including fluidics and aerodynamics, velocity and acceleration vector fields are used as fundamental tools to describe motion or external force of objects, respectively. In analogy, RNA velocity or protein accelerations estimated from single cells can be regarded as samples in the velocity (La Manno et al. 2018) or acceleration vector field (Gorin, Svensson, and Pachter 2019). In general, a vector field can be defined as a vector-valued function f that maps any points (or cells’ expression state) x in a domain Ω with D dimension (or the gene expression system with D transcripts / proteins) to a vector y (for example, the velocity or acceleration for different genes or proteins), that is f(x) = y. 

To formally define the problem of velocity vector field learning, we consider a set of measured cells with pairs of current and estimated future expression states. The difference between the predicted future state and current state for each cell corresponds to the velocity. We suppose that the measured single-cell velocity is sampled from a smooth, differentiable vector field f that maps from xi to yi on the entire domain. Normally, single cell velocity measurements are results of biased, noisy and sparse sampling of the entire state space, thus the goal of velocity vector field reconstruction is to robustly learn a mapping function f that outputs yj given any point xj on the domain based on the observed data with certain smoothness constraints (Jiayi Ma et al. 2013). Under ideal scenario, the mapping function f should recover the true velocity vector field on the entire domain and predict the true dynamics in regions of expression space that are not sampled. The discussion introduced above is based on velocity vector field but it can be similarly extended into acceleration vector field (Gorin, Svensson, and Pachter 2019), etc. To reconstruct vector field function in dynamo, we use::

	dyn.tl.VectorField(adata, **params)


Map potential landscape
'''''''''''''''''''''''
The concept of potential landscape is widely appreciated across various biological disciplines, for example the adaptive landscape in population genetics, protein-folding funnel landscape in biochemistry, epigenetic landscape in developmental biology. In the context of cell fate transition, for example differentiation, carcinogenesis, etc, a potential landscape will not only offers an intuitive description of the global dynamics of the biological process but also provides key insights to understand the multi-stability and transition rate between different cell types and quantify the optimal path of cell fate transition. Because the conventional definition of potential function in physics is not applicable to open biological system, we and others proposed different strategies to provide a biological equivalent by decomposing the stochastic differential equations into either the gradient or the _curl component and uses the gradient part to define the potential. While it is still impossible to obtain the analytical form of the potential function, we are able to use an efficient numerical algorithm we recently developed to map the global potential landscape. This approach uses a least action method under the A-type stochastic integration(Shi et al. 2012), a method that reconciles the “noise effects” resulting from using different stochastic integration methods, for example the predominant Ito or Stratonovich method, which leads to the incompatibility of fixed points under different noise levels.

To globally map the potential landscape Ψ(x), the numerical algorithm (Tang et al. 2017) takes the vector field function f(x)(in terms of kernel basis in our case):: 

	dyn.tl.Potential(adata, **params) 


Visualization
'''''''''''''
In two or three dimensions, a streamline plot can be used to visualize the paths of cells will follow if released in different regions of the gene expression state space under a steady flow field. Another more intuitive way to visualize the structure of vector field is the so called line integral convolution method or LIC (Cabral and Leedom 1993), which works by adding random black-and-white paint sources on the vector field and letting the flowing particle on the vector field picking up some texture to ensure the same streamline having similar intensity. Although we have not discussed in this study, with vector field that changes over time, similar methods, for example, streakline, pathline, timeline, etc. can be used to visualize the evolution of single cell or cell populations.

In dynamo, we use the yt_'s annotate_line_integral_convolution function visualize the vector field reconstructed from dynamo::

	dyn.pl.plot_LIC(adata, **params)

 
.. _`Install Dynamo`: https://github.com/aristoteleo/dynamo-release 
.. _`NASC-seq analysis pipeline`: https://github.com/sandberg-lab/NASC-seq
.. _`velocyto command line interface`: http://velocyto.org/velocyto.py/tutorial/cli.html
.. _`bustool from Pachter lab`:  http://pachterlab.github.io/kallistobus
.. _preprint: https://www.biorxiv.org/content/10.1101/696724v1
.. _yt: https://github.com/yt-project/yt
