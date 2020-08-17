from ..tools.cell_vectors import cell_velocities
from .topography import VectorField
from .vector_calculus import acceleration

def cell_accelerations(adata,
                       vf_basis='pca',
                       basis='umap',
                       enforce=True,
                       preserve_len=True,
                       other_kernels_dict={},
                       **kwargs):
    """Compute RNA acceleration field via reconstructed vector field and project it to low dimensional embeddings.

    In classical physics, including fluidics and aerodynamics, velocity and acceleration vector fields are used as
    fundamental tools to describe motion or external force of objects, respectively. In analogy, RNA velocity or
    accelerations estimated from single cells can be regarded as samples in the velocity (La Manno et al. 2018) or
    acceleration vector field (Gorin, Svensson, and Pachter 2019). In general, a vector field can be defined as a
    vector-valued function f that maps any points (or cells’ expression state) x in a domain Ω with D dimension (or the
    gene expression system with D transcripts / proteins) to a vector y (for example, the velocity or acceleration for
    different genes or proteins), that is f(x) = y.

    In two or three dimensions, a velocity vector field is often visualised as a quiver plot where a collection of arrows
    with a given magnitude and direction is drawn. For example, the velocity estimates of unspliced transcriptome of
    sampled cells projected into two dimensions is drawn to show the prediction of the future cell states in RNA velocity
    (La Manno et al. 2018). During the differentiation process, external signal factors perturb cells and thus change
    the vector field. Since we perform genome-wide profiling of cell states and the experiments performed are often done
    in a short time scale, we assume a constant vector field without loss of generality (See also Discussion). Assuming
    an asymptotic deterministic system, the trajectory of the cells travelling in the gene expression space follows the
    vector field and can be calculated using numerical integration methods, for example Runge-Kutta algorithm. In two or
    three dimensions, a streamline plot can be used to visualize the paths of cells will follow if released in different
    regions of the gene expression state space under a steady flow field. Another more intuitive way to visualize the
    structure of vector field is the so called line integral convolution method or LIC (Cabral and Leedom 1993), which
    works by adding random black-and-white paint sources on the vector field and letting the flowing particle on the
    vector field picking up some texture to ensure the same streamline having similar intensity. Although we have not
    provides such functionalities in dynamo, with vector field that changes over time, similar methods, for example,
    streakline, pathline, timeline, etc. can be used to visualize the evolution of single cell or cell populations.

    Arguments
    ---------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        vf_basis: 'int' (optional, default `pca`)
            The dictionary key that corresponds to the low dimensional embedding where the vector field function
            reconstructed.
        basis: 'int' (optional, default `umap`)
            The dictionary key that corresponds to the reduced dimension in `.obsm` attribute.
        enforce: `bool` (default: `False`)
            Whether to enforce 1) redefining use_for_velocity column in obs attribute;
                               2) recalculation of transition matrix.
        preserve_len: `bool` (default: `True`)
            Whether to preserve the length of high dimension vector length. When set to be True, the length  of low
            dimension projected vector will be proportionally scaled to that of the high dimensional vector. Note that
            when preserve_len is set to be True, the acceleration field may seem to be messy (although the magnitude will
            be reflected) while the trend of acceleration when `preserve_len` is `True` is more clearer but will lose
            information of acceleration magnitude. This is because the acceleration is not directly related to the
            distance of cells in the low embedding space; thus the acceleration direction can be better preserved than
            the magnitude. On the other hand, velocity is more relevant to the distance in low embedding space, so
            preserving magnitude and direction of velocity vector in low dimension can be more easily achieved.
        other_kernels_dict: `dict` (default: `{}`)
            A dictionary of paramters that will be passed to the cosine/correlation kernel.

    Returns
    -------
        Adata: :class:`~anndata.AnnData`
            Returns an updated `~anndata.AnnData` with transition_matrix and projected embedding of high dimension velocity
            vectors in the existing embeddings of current cell state, calculated using either the Itô kernel method
            (default) or the diffusion approximation or the method from (La Manno et al. 2018).
    """

    if 'velocity_' + vf_basis not in adata.obsm.keys():
        cell_velocities(adata, basis=vf_basis)

    if 'VecFld_' + vf_basis not in adata.uns_keys():
        VectorField(adata, basis=vf_basis)

    if 'acceleration_' + vf_basis not in adata.obsm.keys():
        acceleration(adata, basis=vf_basis)

    X = adata.obsm['X_' + vf_basis]
    V_mat = adata.obsm['acceleration_' + vf_basis]
    X_embedding = adata.obsm['X_' + basis]

    if basis != vf_basis and vf_basis.lower() not in ['umap', 'tsne', 'trimap', 'ddtree', 'diffusion_map']:
        cell_velocities(
            adata,
            X=X,
            V_mat=V_mat,
            X_embedding=X_embedding,
            basis=basis,
            enforce=enforce,
            key='acceleration',
            preserve_len=preserve_len,
            other_kernels_dict=other_kernels_dict,
            **kwargs
        )

