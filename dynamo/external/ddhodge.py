# YEAR: 2019
# COPYRIGHT HOLDER: ddhodge

# Code adapted from https://github.com/kazumits/ddhodge.

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import inv, qr
from itertools import combinations
from igraph import Graph
from ..tools.scVectorField import vector_field_function, graphize_vecfld

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


def potential(g):
    """potential is related to the instrinsic time. Note that the returned value from this function is the negative of
    potential. Thus small potential can related to smaller intrinsic time and vice versa."""

    div_neg = -div(g)
    g.to_undirected()
    L = np.array(g.laplacian())
    Q, R = qr(L)
    p = inv(R).dot(Q.T).dot(div_neg)

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


def ddhoge(adata,
           X_data=None,
           layer=None,
           basis="umap",
           dims=None,
           n=30,
           VecFld=None,
           build_graph_method='graphize_vecfld'):
    """Modeling Latent Flow Structure using Hodge Decomposition.

    Integration with curl-free/divergence-free vector field reconstruction.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        X_data: `np.ndarray` (default: `None`)
            The user supplied expression (embedding) data that will be used for calculating diffusion matrix directly.
        layer: `str` or None (default: None)
            Which layer of the data will be used for diffusion matrix calculation.
        basis: `str` (default: `umap`)
            Which basis of the data will be used for diffusion matrix calculation.
        dims: `list` or None (default: `None`)
            The list of dimensions that will be selected for diffusion matrix calculation. If `None`, all dimensions will be used.
        n: `int` (default: `10`)
            Number of nearest neighbors when the nearest neighbor graph is not included.
        VecFld: `dictionary` or None (default: None)
            The reconstructed vector field function.

        Returns
    -------
        adata: :class:`~anndata.AnnData`
            `AnnData` object that is updated with the `ddhodge` key in the `obsp` attribute which to adjacency matrix that
             corresponds to the sparse diffusion graph. Two columns `potential` and `divergence` corresponds to the potential
             and divergence for each cell will also be added.
"""

    X_data = adata.obsm['X_' + basis] if X_data is None else X_data
    if VecFld is None:
        VecFld_key = 'VecFld' if basis is None else "VecFld_" + basis
        if VecFld_key not in adata.uns.keys():
            raise ValueError(
                f'Vector field function {VecFld_key} is not included in the adata object!'
                f'Try firstly running dyn.tl.VectorField(adata, basis={basis})')
        VecFld = adata.uns[VecFld_key]['VecFld']

    func = lambda x: vector_field_function(x, VecFld, dim=dims)

    if build_graph_method:
        neighbor_key = "neighbors" if layer is None else layer + "_neighbors"
        if neighbor_key not in adata.uns_keys():
            Idx = None
        else:
            neighbors = adata.uns[neighbor_key]["connectivities"]
            Idx = neighbors.tolil().rows

        adj_mat = graphize_vecfld(func, X_data, nbrs_idx=Idx, k=n, distance_free=True, n_int_steps=20)

    else:
        if "transition_matrix" not in adata.uns.keys():
            raise Exception(f"Your adata doesn't have transition matrix created. You need to first "
                            f"run dyn.tl.cell_velocity(adata) to get the transition before running"
                            f" this function.")

        adj_mat = adata.uns["transition_matrix"]

    g = build_graph(adj_mat)

    adata.obsp['ddhodge'] = adj_mat
    adata.obs['potential'], adata.obs['divergence'] = potential(g), div(g)
