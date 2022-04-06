# YEAR: 2019
# COPYRIGHT HOLDER: ddhodge

# Code adapted from https://github.com/kazumits/ddhodge.

from itertools import combinations

import numpy as np
from igraph import Graph
from scipy.linalg import qr
from scipy.sparse import csr_matrix


def gradop(g):
    e = np.array(g.get_edgelist())
    ne = g.ecount()
    i, j, x = np.tile(range(ne), 2), e.T.flatten(), np.repeat([-1, 1], ne)

    return csr_matrix((x, (i, j)), shape=(ne, g.vcount()))


def divop(g):
    return -gradop(g).T


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
            cc[i] = [
                1,
                1 if e[0, 1] == e[1, 0] or e[0, 0] == e[1, 1] else -1,
                1 if e[0, 1] == e[2, 0] or e[0, 0] == e[2, 1] else -1,
            ]

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


def grad(g, div_neg=None):
    return gradop(g).dot(potential(g, div_neg))


def div(g):
    """calculate divergence for each cell. negative values correspond to potential sink while positive corresponds to
    potential source. https://en.wikipedia.org/wiki/Divergence"""
    weight = np.array(g.es.get_attribute_values("weight"))
    return divop(g).dot(weight)


def curl(g):
    """calculate curl for each cell. On 2d, negative values correspond to clockwise rotation while positive corresponds
    to anticlockwise rotation.
    https://www.khanacademy.org/math/multivariable-calculus/greens-theorem-and-stokes-theorem/formal-definitions-of
    -divergence-and-curl/a/defining-curl"""

    weight = np.array(g.es.get_attribute_values("weight"))
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
    # sources, targets = adj_mat.nonzero()
    # edgelist = list(zip(sources.tolist(), targets.tolist()))
    # g = Graph(edgelist, edge_attrs={"weight": adj_mat.data.tolist()}, directed=True)
    g = Graph.Weighted_Adjacency(adj_mat)

    return g
