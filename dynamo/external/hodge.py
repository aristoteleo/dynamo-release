# YEAR: 2019
# COPYRIGHT HOLDER: ddhodge

# Code adapted from https://github.com/kazumits/ddhodge.

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import qr
from itertools import combinations
from igraph import Graph
from ..vectorfield.scVectorField import graphize_vecfld
from ..vectorfield.utils_vecCalc import vecfld_from_adata, vector_field_function
from ..tools.sampling import trn, sample_by_velocity

def gradop(g):
    e = np.array(g.get_edgelist())
    ne = g.ecount()
    i, j, x = np.tile(range(ne), 2), e.T.flatten(), np.repeat([-1, 1], ne)

    return csr_matrix((x, (i, j)), shape=(ne, g.vcount()))


def divop(g):
    return - gradop(g).T


def curlop(g):
    triv = np.array(g.cliques(min=3, max=3))
    ntri = triv.shape[0]

    if ntri == 1:
        return np.zeros((0, g.ecount())) * np.nan
    else:
        trie = np.zeros_like(triv)
        for i, x in enumerate(triv):
            trie[i] = g.get_eids(path=np.hstack((x, x[0])), directed=False)

        edges = np.array(g.get_edgelist())
        cc = np.zeros_like(trie)
        for i, x in enumerate(trie):
            e = edges[x]
            cc[i] = [1,
                     1 if e[0, 1] == e[1, 0] or e[0, 0] == e[1, 1] else -1,
                     1 if e[0, 1] == e[2, 0] or e[0, 0] == e[2, 1] else -1]

    i, j, x = np.repeat(range(ntri), 3), trie.flatten(), cc.flatten()

    return csr_matrix((x, (i, j)), shape=(ntri, g.ecount()))


def laplacian0(g):
    mat = gradop(g)

    return mat.T.dot(mat)


def laplacian1(g):
    cur_mat, grad_mat = curlop(g), gradop(g)

    return cur_mat.T.dot(cur_mat) - grad_mat.dot(grad_mat.T)


def potential(g, div_neg=None):
    """potential is related to the intrinsic time. Note that the returned value from this function is the negative of
    potential. Thus small potential is related to smaller intrinsic time and vice versa."""

    div_neg = -div(g) if div_neg is None else div_neg
    g_undirected = g.copy()
    g_undirected.to_undirected()
    L = np.array(g_undirected.laplacian())
    Q, R = qr(L)
    p = np.linalg.pinv(R).dot(Q.T).dot(div_neg)

    res = p - p.min()
    return res


def grad(g, tol=1e-7):
    return gradop(g).dot(potential(g, tol))


def div(g):
    """calculate divergence for each cell. negative values correspond to potential sink while positive corresponds to
    potential source. https://en.wikipedia.org/wiki/Divergence"""
    weight = np.array(g.es.get_attribute_values('weight'))
    return divop(g).dot(weight)


def curl(g):
    """calculate curl for each cell. On 2d, negative values correspond to clockwise rotation while positive corresponds to
    anticlockwise rotation. https://www.khanacademy.org/math/multivariable-calculus/greens-theorem-and-stokes-theorem/formal-definitions-of-divergence-and-curl/a/defining-curl"""

    weight = np.array(g.es.get_attribute_values('weight'))
    return curlop(g).dot(weight)


def triangles(g):
    cliques = g.cliques(min=3, max=3)
    result = [0] * g.vcount()
    for i, j, k in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
    return result


def _triangles(g):
    result = [0] * g.vcount()
    adjlist = [set(neis) for neis in g.get_adjlist()]
    for vertex, neis in enumerate(adjlist):
        for nei1, nei2 in combinations(neis, 2):
            if nei1 in adjlist[nei2]:
                result[vertex] += 1
    return result


def build_graph(adj_mat):
    """build sparse diffusion graph. The adjacency matrix need to preserves divergence."""
    sources, targets = adj_mat.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    g = Graph(edgelist, edge_attrs={'weight': adj_mat.data.tolist()}, directed=True)

    return g


def ddhodge(adata,
            X_data=None,
            layer=None,
            basis="pca",
            n=30,
            VecFld=None,
            adjmethod='graphize_vecfld',
            distance_free=False,
            n_downsamples=5000,
            up_sampling=True,
            sampling_method='velocity',
            seed=19491001,
            enforce=False,
            cores=1):
    """Modeling Latent Flow Structure using Hodge Decomposition based on the creation of sparse diffusion graph from the
    reconstructed vector field function. This method is relevant to the curl-free/divergence-free vector field
    reconstruction.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        X_data: `np.ndarray` (default: `None`)
            The user supplied expression (embedding) data that will be used for graph hodege decomposition directly.
        layer: `str` or None (default: None)
            Which layer of the data will be used for graph Hodge decomposition.
        basis: `str` (default: `pca`)
            Which basis of the data will be used for graph Hodge decomposition.
        n: `int` (default: `10`)
            Number of nearest neighbors when the nearest neighbor graph is not included.
        VecFld: `dictionary` or None (default: None)
            The reconstructed vector field function.
        adjmethod: `str` (default: `graphize_vecfld`)
            The method to build the ajacency matrix that will be used to create the sparse diffusion graph, can be either
            "naive" or "graphize_vecfld". If "naive" used, the transition_matrix that created during vector field projection
            will be used; if "graphize_vecfld" used, a method that guarantees the preservance of divergence will be used.
        n_downsamples: `int` (default: `5000`)
            Number of cells to downsample to if the cell number is large than this value. Three downsampling methods are
            available, see `sampling_method`.
        up_sampling: `bool` (default: `True`)
            Whether to assign calculated potential, curl and divergence to cells not sampled based on values from their
            nearest sampled cells.
        sampling_method: `str` (default: `random`)
            Methods to downsample datasets to facilitate calculation. Can be one of {`random`, `velocity`, `trn`}, each
            corresponds to random sampling, velocity magnitude based and topology representing network based sampling.
        seed : int or 1-d array_like, optional (default: `0`)
            Seed for RandomState. Must be convertible to 32 bit unsigned integers. Used in sampling control points. Default
            is to be 0 for ensure consistency between different runs.
        enforce: `bool` (default: `False`)
            Whether to enforce the calculation of adjacency matrix for estimating potential, curl, divergence for each
            cell.
        cores: `int` (default: 1):
            Number of cores to run the graphize_vecfld function. If cores is set to be > 1, multiprocessing will be used
            to parallel the graphize_vecfld calculation.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            `AnnData` object that is updated with the `ddhodge` key in the `obsp` attribute which to adjacency matrix that
             corresponds to the sparse diffusion graph. Two columns `potential` and `divergence` corresponds to the potential
             and divergence for each cell will also be added.
"""

    prefix = '' if basis is None else basis + '_'
    to_downsample = adata.n_obs > n_downsamples

    if VecFld is None:
        VecFld, func = vecfld_from_adata(adata, basis)
    else:
        func = lambda x: vector_field_function(x, VecFld)

    if X_data is None:
        X_data_full = VecFld['X'].copy()
    else:
        if X_data.shape[0] != adata.n_obs:
            raise ValueError(f"The X_data you provided doesn't correspond to exactly {adata.n_obs} cells")
        X_data_full = X_data.copy()


    if to_downsample:
        if sampling_method == 'trn':
            cell_idx = trn(X_data_full, n_downsamples)
        elif sampling_method == 'velocity':
            np.random.seed(seed)
            cell_idx = sample_by_velocity(func(X_data_full), n_downsamples)
        elif sampling_method == 'random':
            np.random.seed(seed)
            cell_idx = np.random.choice(np.arange(adata.n_obs), n_downsamples)
        else:
            raise ImportError(f"sampling method {sampling_method} is not available. Only `random`, `velocity`, `trn` are"
                              f"available.")
    else:
        cell_idx = np.arange(adata.n_obs)

    X_data = X_data_full[cell_idx, :]
    adata_ = adata[cell_idx].copy()

    if prefix + 'ddhodge' in adata_.obsp.keys() and not enforce and not to_downsample:
        adj_mat = adata_.obsp[prefix + 'ddhodge']
    else:
        if (adjmethod == 'graphize_vecfld'):
            neighbor_key = "neighbors" if layer is None else layer + "_neighbors"
            if neighbor_key not in adata_.uns_keys() or to_downsample:
                Idx = None
            else:
                conn_key = "connectivities" if layer is None else layer + "_connectivities"
                neighbors = adata_.obsp[conn_key]
                Idx = neighbors.tolil().rows

            adj_mat, nbrs = graphize_vecfld(func, X_data, nbrs_idx=Idx, k=n, distance_free=distance_free, n_int_steps=20,
                                      cores=cores)
        elif adjmethod == 'naive':
            if "transition_matrix" not in adata_.uns.keys():
                raise Exception(f"Your adata doesn't have transition matrix created. You need to first "
                                f"run dyn.tl.cell_velocity(adata) to get the transition before running"
                                f" this function.")

            adj_mat = adata_.uns["transition_matrix"][cell_idx, cell_idx]
        else:
            raise ValueError(f"adjmethod can be only one of {'naive', 'graphize_vecfld'}")

    g = build_graph(adj_mat)

    if (prefix + 'ddhodge' not in adata.obsp.keys() or enforce) and not to_downsample:
        adata.obsp[prefix + 'ddhodge'] = adj_mat

    ddhodge_div = div(g)
    potential_ = potential(g, - ddhodge_div)

    if up_sampling and to_downsample:
        query_idx = list(set(np.arange(adata.n_obs)).difference(cell_idx))
        query_data = X_data_full[query_idx, :]
        
        if hasattr(nbrs, 'kneighbors'): 
            dist, nbrs_idx = nbrs.kneighbors(query_data)
        elif hasattr(nbrs, 'query'): 
            nbrs_idx, dist = nbrs.query(query_data, k=nbrs.n_neighbors)

        k = nbrs_idx.shape[1]
        row, col = np.repeat(np.arange(len(query_idx)), k), nbrs_idx.flatten()
        W = csr_matrix((np.repeat(1/k, len(row)), (row, col)), shape=(len(query_idx), len(cell_idx)))

        query_data_div, query_data_potential = W.dot(ddhodge_div), W.dot(potential_)
        adata.obs[prefix + 'ddhodge_sampled'], adata.obs[prefix + 'ddhodge_div'], adata.obs[prefix + 'potential'] = False, 0, 0
        adata.obs.loc[adata.obs_names[cell_idx], prefix + 'ddhodge_sampled'] = True
        adata.obs.loc[adata.obs_names[cell_idx], prefix + 'ddhodge_div'] = ddhodge_div
        adata.obs.loc[adata.obs_names[cell_idx], prefix + 'potential'] = potential_
        adata.obs.loc[adata.obs_names[query_idx], prefix + 'ddhodge_div'] = query_data_div
        adata.obs.loc[adata.obs_names[query_idx], prefix + 'potential'] = query_data_potential
    else:
        adata.obs[prefix + 'ddhodge_div'] = ddhodge_div
        adata.obs[prefix + 'ddhodge_potential'] = potential_
