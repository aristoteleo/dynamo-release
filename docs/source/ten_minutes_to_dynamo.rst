10 minutes to dynamo
--------------------

Welcome to dynamo!

Dynamo is a computational framework that includes an inclusive model of expression dynamics with scSLAM-seq / multiomics, vector field reconstruction and potential landscape mapping.


How to install
^^^^^^^^^^^^^^
Dynamo requires Python 3.6 or later.

Dynamo now has been released to PyPi, you can install the PyPi version via::

    pip install dynamo-release

To install the newest version of dynamo, you can git clone our repo and then use::

	pip install directory_to_dynamo_release_repo/

Alternatively, you can install dynamo from source using::

    pip install git+https://github.com/aristoteleo/dynamo-release

To process the scSLAM-seq data, please refer to the `NASC-seq analysis pipeline`_. We are also working on a command line tool for this too. For processing splicing data, you
can either use the `velocyto command line interface`_ or the `bustool from Pachter lab`_.

Architecture of dynamo
^^^^^^^^^^^^^^^^^^^^^^
.. image:: https://raw.githubusercontent.com/Xiaojieqiu/jungle/master/test.png

Basic Usage
^^^^^^^^^^^
Import dynamo as::

    import dynamo as dyn

We provide a few nice visualization defaults for different purpose::

    dyn.configuration.set_figure_params('dynamo', background='white') # jupter notebooks
    dyn.configuration.set_figure_params('dynamo', background='black') # presentation
    dyn.configuration.set_pub_style() # manuscript

Typical workflow
^^^^^^^^^^^^^^^^
.. image:: https://raw.githubusercontent.com/Xiaojieqiu/jungle/master/dynamo_workflow.png

A typical workflow in dynamo is similar to most of other single cell analysis toolkits (Scanpy, Monocle or Seurat), including steps like loading data (``dyn.read*``), preprocessing (``dyn.pp.*``), tool analysis (``dyn.tl.*``) and plotting (``dyn.pl.*``). Other steps that are unique to dynamo include, conventional single cell RNA-seq (scRNA-seq) modeling (``dyn.csc.*``), time-resolved metabolic labeling based single cell RNA-seq (scRNA-seq) modeling (``dyn.tsc.*``), vector field analysis (``dyn.vf.*``), cell fate prediction (``dyn.pd.*``), creating movie of cell fate commitment (``dyn.mv.*``), simulation (``dyn.sim.*``), integration with external toolkit built by us or others (``dyn.ext.*``).

Load data
''''''''''
Dynamo relies on anndata for data IO. You can read your own data via ``read``, ``read_loom``, ``read_h5ad``, ``read_h5`` or ``load_NASC_seq``, etc::

    adata = dyn.read(filename)

Dynamo also comes with a few builtin sample datasets. For example, you can load the Dentate Gyrus example dataset::

    adata = dyn.sample_data.DentateGyrus()

There are many sample datasets available. You can check our tutorials for other available datasets via ``dyn.sample_data.*``.

Preprocess data
'''''''''''''''
Then you are ready to performs preprocessing of the data. You can run the `recipe_monocle` function that uses similar strategy from Monocle 3 to normalize all datasets in different layers (the spliced and unspliced or new (metabolic labelled) and total mRNAs or others), followed by feature selection and PCA dimension reduction.::

    dyn.pp.recipe_monocle(adata)

Learn dynamics
''''''''''''''
Next you will want to estimate the kinetic parameters of expression dynamics and then learn the velocity values for all genes that pass some filters (selected feature genes, by default) across cells. The ``dyn.tl.dynamics`` does all the hard work for you. It checks the data you have and determine the experimental type automatically, either the conventional scRNA-seq, kinetics, degradation or one-shot single-cell metabolic labelling experiment or the CITE-seq or REAP-seq co-assay, etc. Then it learns the velocity for each feature gene using either the original deterministic model based on a steady-state assumption from the seminal RNA velocity work or a few new methods, including the stochastic or negative binomial method for conventional scRNA-seq or kinetic, degradation or one-shot models for metabolic labeling based scRNA-seq. Those later methods are based on moment equations which basically considers both mean and uncentered variance of gene expression. The second model requires the calculation of the first and second moment of the expression data that is based on a nearest neighbours graph, constructed in the reduced PCA space from the spliced or total mRNA expression.::

    dyn.tl.dynamics(adata)

which implicitly calls::

    dyn.tl.moments(adata)

Kinetic estimation of the conventional scRNA-seq and metabolic labeling based scRNA-seq is often tricky. You can evaluate the confidence of gene-wise velocity via::

    dyn.tl.gene_wise_confidence(adata)

Dimension reduction
'''''''''''''''''''
By default, we use ``umap`` algorithm for dimension reduction.::

    dyn.tl.reduceDimension(adata)

Velocity vectors
''''''''''''''''
We then need to project the velocity vector on low dimensional embedding. To get there, we can either use the default "correlation/cosine kernel" or the novel Itô kernel from us.::

    dyn.tl.cell_velocities(adata)

You can check the confidence of cell-wise velocity via::

    dyn.tl.cell_wise_confidence(adata)

The above functions projects and evaluate velocity vectors on ``umap`` space but you can also operate them on other basis, for example ``pca`` space::

    dyn.tl.cell_velocities(adata, basis='pca')
    dyn.tl.cell_wise_confidence(adata, basis='pca')

Obviously dynamo doesn't want to stop here. The really exciting part of dynamo lays in the fact that it learns a ``functional form of vector field`` in the full transcriptomic space which can be then used to the cell fate potential.

Vector field reconstruction
'''''''''''''''''''''''''''
In classical physics, including fluidics and aerodynamics, velocity and acceleration vector fields are used as fundamental tools to describe motion or external force of objects, respectively. In analogy, RNA velocity or protein accelerations estimated from single cells can be regarded as sparse samples in the velocity (La Manno et al. 2018) or acceleration vector field (Gorin, Svensson, and Pachter 2019). In general, a vector field can be defined as a vector-valued function f that maps any points (or cells’ expression state) x in a domain Ω with D dimension (or the gene expression system with D transcripts / proteins) to a vector y (for example, the velocity or acceleration for different genes or proteins), that is f(x) = y.

To formally define the problem of velocity vector field learning, we consider a set of measured cells with pairs of current and estimated future expression states. The difference between the predicted future state and current state for each cell corresponds to the velocity. We suppose that the measured single-cell velocity is sampled from a smooth, differentiable vector field f that maps from xi to yi on the entire domain. Normally, single cell velocity measurements are results of biased, noisy and sparse sampling of the entire state space, thus the goal of velocity vector field reconstruction is to robustly learn a mapping function f that outputs yj given any point xj on the domain based on the observed data with certain smoothness constraints (Jiayi Ma et al. 2013). Under ideal scenario, the mapping function f should recover the true velocity vector field on the entire domain and predict the true dynamics in regions of expression space that are not sampled. To reconstruct vector field function in dynamo, we use::

	dyn.tl.VectorField(adata)

By default, it learns the vector field in the `pca` space.

Vector field reconstruction
'''''''''''''''''''''''''''
Since we learn the vector function of the data, we can then characterize the topology of the full vector field space. For example, we are able to identify the fixed points (attractor/saddles, etc.) which may corresponds to terminal cell types or progenitors, nullcline, separatrices of a recovered dynamic system, which may formally define the dynamical behaviour or the boundary of cell types in gene expression space.::

    dyn.tl.topography(adata, basis='umap')

Map potential landscape
'''''''''''''''''''''''
The concept of potential landscape is widely appreciated across various biological disciplines, for example the adaptive landscape in population genetics, protein-folding funnel landscape in biochemistry, epigenetic landscape in developmental biology. In the context of cell fate transition, for example differentiation, carcinogenesis, etc, a potential landscape will not only offers an intuitive description of the global dynamics of the biological process but also provides key insights to understand the multi-stability and transition rate between different cell types and quantify the optimal path of cell fate transition.

Because the conventional definition of potential function in physics is not applicable to open biological system, in dynamo we provided several ways to quantify the potential of single cells by decomposing the vector field into gradient or curl parts. The recommended method is built on the Hodge decomposition on simplicial complexes (a sparse directional graph) constructed based on the learned vector field function that provides fruitful analogy of gradient, curl and harmonic (cyclic) flows on manifold::

	dyn.ext.ddhoge(adata)

In addition, we and others proposed different strategies to decompose the stochastic differential equations into either the gradient or the _curl component and uses the gradient part to define the potential. While it is still impossible to obtain the analytical form of the potential function, we are able to use an efficient numerical algorithm we recently developed to map the global potential landscape. This approach uses a least action method under the A-type stochastic integration(Shi et al. 2012), a method that reconciles the “noise effects” resulting from using different stochastic integration methods, for example the predominant Ito or Stratonovich method, which leads to the incompatibility of fixed points under different noise levels.

To globally map the potential landscape Ψ(x), the numerical algorithm (Tang et al. 2017) takes the vector field function f(x)(in terms of kernel basis in our case):: 

	dyn.tl.potential(adata)

Visualization
'''''''''''''
In two or three dimensions, a streamline plot can be used to visualize the paths of cells will follow if released in different regions of the gene expression state space under a steady flow field. Another more intuitive way to visualize the structure of vector field is the so called line integral convolution method or LIC (Cabral and Leedom 1993), which works by adding random black-and-white paint sources on the vector field and letting the flowing particle on the vector field picking up some texture to ensure the same streamline having similar intensity. Although we have not discussed in this study, with vector field that changes over time, similar methods, for example, streakline, pathline, timeline, etc. can be used to visualize the evolution of single cell or cell populations.

In dynamo, we use the yt_'s annotate_line_integral_convolution function visualize the vector field reconstructed from dynamo::

    dyn.pl.cell_wise_vectors(adata, color=gene_list, ncols=3, method='SparseVFC')
    dyn.pl.grid_vectors(adata, color=gene_list, ncols=3, method='SparseVFC')
    dyn.pl.stremline_plot(adata, color=gene_list, ncols=3, method='SparseVFC')
    dyn.pl.line_integral_conv(adata)

To visualize the topography of the learnt vector field, we provide the ``dyn.pl.topography`` function to visualize the structure of the 2D vector fields.::

    dyn.pl.topography(adata)

Note that if you only visualize one item for each plot function, you can combine different types of dynamo plots together::

    import matplotlib.pyplot as plt
    fig1, f1_axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True, figsize=(12, 8))
    f1_axes
    f1_axes[0, 0] = dyn.pl.cell_wise_vectors(adata, color='umap_ddhodge_potential', pointsize=0.1, alpha = 0.7, ax=f1_axes[0, 0], quiver_length=6, quiver_size=6, save_show_or_return='return')
    f1_axes[0, 1] = dyn.pl.grid_vectors(adata, color='speed_umap', ax=f1_axes[0, 1], quiver_length=12, quiver_size=12, save_show_or_return='return')
    f1_axes[1, 0] = dyn.pl.streamline_plot(adata, color='divergence_pca', ax=f1_axes[1, 0], save_show_or_return='return')
    f1_axes[1, 1] = dyn.pl.streamline_plot(adata, color='acceleration_umap', ax=f1_axes[1, 1], save_show_or_return='return')
    plt.show()

Comparability
^^^^^^^^^^^^^
Dynamo is fully compatible with velocyto, scanpy and scvelo. So you can use your loom or annadata object as input for dynamo. The velocity vector samples estimated from either velocyto or scvelo can be also directly used to reconstruct the functional form of vector field
and the potential landscape in the entire expression space.

.. _`Install Dynamo`: https://github.com/aristoteleo/dynamo-release 
.. _`NASC-seq analysis pipeline`: https://github.com/sandberg-lab/NASC-seq
.. _`velocyto command line interface`: http://velocyto.org/velocyto.py/tutorial/cli.html
.. _`bustool from Pachter lab`:  http://pachterlab.github.io/kallistobus
.. _preprint: https://www.biorxiv.org/content/10.1101/696724v1
.. _yt: https://github.com/yt-project/yt
