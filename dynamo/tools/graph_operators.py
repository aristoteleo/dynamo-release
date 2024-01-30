""" This file implements graph operators using Graph object from iGraph as input.

YEAR: 2019
COPYRIGHT HOLDER: ddhodge

Code adapted from https://github.com/kazumits/ddhodge.
"""
from typing import List, Optional, Union

from itertools import combinations

import numpy as np
from igraph import Graph
from scipy.linalg import qr
from scipy.sparse import csr_matrix


def potential(g: Graph, div_neg: Optional[csr_matrix] = None) -> np.ndarray:
    """Calculate potential for each cell.

    The potential is related to the intrinsic time. Note that the returned value from this function is the negative of
    potential. Thus, small potential is related to smaller intrinsic time and vice versa.

    Args:
        g: Graph object.
        div_neg: Negative divergence. If None, it will be calculated from the graph.

    Returns:
        An array representing the potential.
    """

    div_neg = -div(g) if div_neg is None else div_neg
    g_undirected = g.copy()
    g_undirected.to_undirected()
    L = np.array(g_undirected.laplacian())
    Q, R = qr(L)
    p = np.linalg.pinv(R).dot(Q.T).dot(div_neg)

    res = p - p.min()
    return res


def grad(g: Graph, div_neg: Optional[csr_matrix] = None) -> np.ndarray:
    """Compute the gradient of a potential field on a graph.

    The gradient of a potential field on a graph represents the rate of change of the potential at each vertex. It is
    obtained by multiplying the gradient operator with the potential field.

    Args:
        g: Graph object.
        div_neg: Negative divergence. If None, it will be calculated from the graph.

    Returns:
        An array representing the gradient.
    """
    return gradop(g).dot(potential(g, div_neg))


def gradop(g: Graph) -> csr_matrix:
    """Compute the gradient operator for a graph.

    Args:
        g: Graph object.

    Returns:
        Gradient operator as a sparse matrix.
    """
    e = np.array(g.get_edgelist())
    ne = g.ecount()
    i, j, x = np.tile(range(ne), 2), e.T.flatten(), np.repeat([-1, 1], ne)

    return csr_matrix((x, (i, j)), shape=(ne, g.vcount()))


def div(g: Graph) -> np.ndarray:
    """Calculate divergence for each cell.

    Negative values correspond to potential sink while positive corresponds to potential source.
    https://en.wikipedia.org/wiki/Divergence

    Args:
        g: Graph object.

    Returns:
        The divergence of the graph
    """
    weight = np.array(g.es.get_attribute_values("weight"))
    return divop(g).dot(weight)


def divop(g: Graph) -> csr_matrix:
    """Compute the divergence operator for a graph.

    Args:
        g: Graph object.

    Returns:
        Divergence operator as a sparse matrix.
    """
    return -gradop(g).T


def curl(g: Graph) -> np.ndarray:
    """Calculate curl for each cell.

    On 2d, negative values correspond to clockwise rotation while positive corresponds to anticlockwise rotation.
    https://www.khanacademy.org/math/multivariable-calculus/greens-theorem-and-stokes-theorem/formal-definitions-of
    -divergence-and-curl/a/defining-curl

    Args:
        g: Graph object.

    Returns:
        The curl of the graph.
    """

    weight = np.array(g.es.get_attribute_values("weight"))
    return curlop(g).dot(weight)


def curlop(g: Graph) -> csr_matrix:
    """Compute the curl operator for a graph.

    Args:
        g: Graph object.

    Returns:
        Curl operator as a sparse matrix.
    """
    triv = np.array(g.cliques(min=3, max=3))
    ntri = triv.shape[0]

    if ntri == 1:
        return np.zeros((0, g.ecount())) * np.nan
    else:
        trie = np.zeros_like(triv)
        for i, x in enumerate(triv):
            trie[i] = g.get_eids(pairs=[[x[0], x[1]], [x[1], x[2]], [x[2], x[0]]], directed=False)

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


def laplacian0(g: Graph) -> csr_matrix:
    """Compute the Laplacian for a graph as the square of the gradient operator.

    Args:
        g: Graph object.

    Returns:
        Laplacian as a sparse matrix.
    """
    mat = gradop(g)

    return mat.T.dot(mat)


def laplacian1(g: Graph) -> csr_matrix:
    """Compute the Laplacian for a graph by incorporating both curl and gradient operations.

    Args:
        g: Graph object.

    Returns:
        Laplacian as a sparse matrix.
    """
    cur_mat, grad_mat = curlop(g), gradop(g)

    return cur_mat.T.dot(cur_mat) - grad_mat.dot(grad_mat.T)


def triangles(g: Graph) -> List[int]:
    """Count the number of triangles each vertex participates in within a graph using cliques. A triangle is a cycle of
    length 3 in an undirected graph.

    Args:
        g: Graph object.

    Returns:
        A list of the number of triangles each vertex participates in.
    """
    cliques = g.cliques(min=3, max=3)
    result = [0] * g.vcount()
    for i, j, k in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
    return result


def _triangles(g: Graph) -> List[int]:
    """Count the number of triangles each vertex participates in within a graph using cliques. A triangle is a cycle of
        length 3 in an undirected graph.

    Args:
        g: Graph object.

    Returns:
        A list of the number of triangles each vertex participates in.
    """
    result = [0] * g.vcount()
    adjlist = [set(neis) for neis in g.get_adjlist()]
    for vertex, neis in enumerate(adjlist):
        for nei1, nei2 in combinations(neis, 2):
            if nei1 in adjlist[nei2]:
                result[vertex] += 1
    return result


def build_graph(adj_mat: Union[csr_matrix, np.ndarray]) -> Graph:
    """bBuild sparse diffusion graph. The adjacency matrix need to preserve divergence.

    Args:
        adj_mat: Adjacency matrix of the graph.

    Returns:
        Graph object.
    """
    # sources, targets = adj_mat.nonzero()
    # edgelist = list(zip(sources.tolist(), targets.tolist()))
    # g = Graph(edgelist, edge_attrs={"weight": adj_mat.data.tolist()}, directed=True)
    g = Graph.Weighted_Adjacency(adj_mat)

    return g
