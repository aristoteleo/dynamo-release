import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import dynamo as dyn
from dynamo.tools.connectivity import (
    generate_neighbor_keys,
    check_and_recompute_neighbors,
    check_neighbors_completeness,
)


def test_calc_1nd_moment():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = dyn.tl.calc_1nd_moment(X, W, normalize_W=False)
    expected_result = np.array([[3, 4], [6, 8], [3, 4]])
    assert np.array_equal(result, expected_result)

    result, normalized_W = dyn.tl.calc_1nd_moment(X, W, normalize_W=True)
    expected_result = np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
    assert np.array_equal(result, expected_result)
    assert np.array_equal(normalized_W,
                          np.array([[0.0, 1, 0.0], [0.5, 0.0, 0.5], [0.0, 1, 0.0]]))


def test_calc_2nd_moment():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Y = np.array([[2, 3], [4, 5], [6, 7]])
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = dyn.tl.calc_2nd_moment(X, Y, W, normalize_W=False, center=False)
    expected_result = np.array([[12, 20], [32, 48], [12, 20]])
    assert np.array_equal(result, expected_result)

    result = dyn.tl.calc_2nd_moment(X, Y, W, normalize_W=True, center=False)
    expected_result = np.array([[12., 20.], [16., 24.], [12., 20.]])
    assert np.array_equal(result, expected_result)


def test_cell_growth_rate(adata):
    dyn.tl.cell_growth_rate(
        adata,
        group="Cell_type",
        source="Unknown",
        target="Unknown",
        clone_column="batch",
    )
    assert "growth_rate" in adata.obs.keys()


@pytest.mark.skip(reason="umap compatability issue with numpy, pynndescent and pytest")
def test_dynamics():
    adata = dyn.sample_data.scNT_seq_neuron_labeling()
    adata.obs['label_time'] = 2  # this is the labeling time

    adata = adata[:, adata.var.activity_genes]
    adata.obs['time'] = adata.obs['time'] / 60

    adata1 = adata.copy()
    preprocessor = dyn.pp.Preprocessor(cell_cycle_score_enable=True)
    preprocessor.preprocess_adata(adata1, recipe='monocle', tkey='label_time', experiment_type='one-shot')
    dyn.tl.dynamics(adata1)
    assert "velocity_N" in adata.layers.keys()

    adata2 = adata.copy()
    preprocessor = dyn.pp.Preprocessor(cell_cycle_score_enable=True)
    preprocessor.preprocess_adata(adata2, recipe='monocle', tkey='label_time', experiment_type='kin')
    dyn.tl.dynamics(adata2)
    assert "velocity_N" in adata.layers.keys()


def test_top_n_markers():
    adata = dyn.sample_data.zebrafish()
    adata = adata[:500, :500].copy()
    dyn.tl.find_group_markers(adata, group="Cell_type")
    top_n_df = dyn.tl.top_n_markers(adata, top_n_genes=1)
    assert type(top_n_df) == pd.DataFrame


def test_sampling():
    arr = np.random.rand(20, 3)
    n = 2
    samples = dyn.tl.sample(arr, n)
    assert samples.shape[0] == n
    assert samples[0] in arr
    assert samples[1] in arr

    V = np.random.rand(20, 3)
    samples = dyn.tl.sample(arr, n, method="velocity", V=V)
    assert samples.shape[0] == n
    assert samples[0] in arr
    assert samples[1] in arr

    samples = dyn.tl.sample(arr, n, method="trn", X=arr)
    assert samples.shape[0] == n
    assert samples[0] in arr
    assert samples[1] in arr

    samples = dyn.tl.sample(arr, n, method="kmeans", X=arr)
    assert samples.shape[0] == n
    assert samples[0] in arr
    assert samples[1] in arr


def test_score_cells(adata):
    scores = dyn.tl.score_cells(adata)
    assert scores.shape[0] == adata.n_obs


# clustering
@pytest.mark.skip(reason="dependency not installed")
def test_simple_cluster_community_adata(adata):
    adata = adata.copy()
    dyn.tl.louvain(adata)
    dyn.tl.leiden(adata)

    try:
        dyn.tl.louvain(adata, directed=True)
    except ValueError as e:
        print("################################################")
        print("PASSED: Value error captured as EXPECTED, Exception info:")
        print(e)
        print("################################################")

    dyn.tl.leiden(adata, directed=True)
    assert np.all(adata.obs["louvain"] != -1)
    assert np.all(adata.obs["leiden"] != -1)

    # dyn.pl.louvain(adata, basis="pca")
    # dyn.pl.leiden(adata, basis="pca")
    # dyn.pl.infomap(adata, basis="pca")


@pytest.mark.skip(reason="dependency not installed")
def test_simple_cluster_field(adata):
    adata = adata.copy()
    dyn.vf.cluster_field(adata, basis="umap", method="louvain")
    dyn.vf.cluster_field(adata, basis="umap", method="leiden")
    assert np.all(adata.obs["louvain"] != -1)
    assert np.all(adata.obs["leiden"] != -1)


@pytest.mark.skip(reason="dependency not installed")
def test_leiden_membership_input(adata):
    adata = adata.copy()
    # somehow this initial membership works before, but not now
    initial_membership = np.random.randint(low=0, high=min(100, len(adata)), size=len(adata), dtype=int)
    dyn.tl.leiden(adata, initial_membership=initial_membership)
    assert np.all(adata.obs["leiden"] != -1)

    initial_membership = np.random.randint(low=0, high=100, size=len(adata), dtype=int)
    dyn.tl.leiden(adata, directed=True, initial_membership=initial_membership)
    assert np.all(adata.obs["leiden"] != -1)


def test_DDRTree_pseudotime(adata):
    import matplotlib.pyplot as plt

    adata = adata.copy()
    dyn.tl.order_cells(adata, basis="umap", maxIter=3, ncenter=10)
    assert "Pseudotime" in adata.obs.keys()

    tree = dyn.tl.construct_velocity_tree(adata)
    assert tree.shape == (10, 10)

    # TODO: enable or delete this after debugging
    # dyn.tl.directed_pg(adata, basis="umap", maxIter=3, ncenter=10)
    # assert np.all(adata.uns["X_DDRTree_pg"] != -1)

    dyn.tl.glm_degs(adata, fullModelFormulaStr="cr(Pseudotime, df=3)")
    assert "glm_degs" in adata.uns.keys()

    dyn.tl.pseudotime_velocity(adata, pseudotime="Pseudotime")
    assert "velocity_S" in adata.layers.keys()

    ax = dyn.pl.plot_dim_reduced_direct_graph(adata, graph=adata.uns["directed_velocity_tree"], save_show_or_return="return")
    assert isinstance(ax, list)


def test_psl():
    arr = np.random.rand(20, 10)
    S, Z = dyn.tl.psl(arr, maxIter=3)

    assert S.shape == (20, 20)
    assert Z.shape == (20, 2)


def test_graph_calculus_and_operators():
    adj = csr_matrix(np.random.rand(10, 10))
    graph = dyn.tools.graph_operators.build_graph(adj)

    res_op = dyn.tools.graph_operators.gradop(graph)
    res_calc = dyn.tools.graph_calculus.gradop(adj)
    assert np.allclose(res_op.A, res_calc.A)

    res_op = dyn.tools.graph_operators.divop(graph)
    res_calc = dyn.tools.graph_calculus.divop(adj)
    assert np.allclose(res_op.A, 2 * res_calc.A)

    res_op = dyn.tools.graph_operators.potential(graph)
    res_calc = dyn.tools.graph_calculus.potential(adj.A, method="lsq")
    assert np.allclose(res_op, res_calc * 2)

    potential_lsq = dyn.tools.graph_calculus.potential(adj.A, method="lsq")
    res_op = dyn.tools.graph_operators.grad(graph)
    res_calc = dyn.tools.graph_calculus.gradient(adj.A, p=potential_lsq)
    assert np.allclose(res_op, 2 * res_calc)

    res_op = dyn.tools.graph_operators.div(graph)
    res_calc = dyn.tools.graph_calculus.divergence(adj.A)
    assert np.allclose(res_op, 2 * res_calc)


def test_calc_laplacian():
    # Test naive weight mode
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    expected_result = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
    assert np.allclose(dyn.tl.graph_calculus.calc_laplacian(W, convention="graph"), expected_result)

    # Test E argument
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    E = np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]])
    expected_result = np.array([[0.25, -0.25, 0], [-0.25, 1.25, -1], [0, -1, 1]])
    assert np.allclose(
        dyn.tl.graph_calculus.calc_laplacian(W, E=E, weight_mode="asymmetric", convention="graph"),
        expected_result,
    )


def test_divergence():
    # Create a test adjacency matrix
    adj = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    M = np.array([[0, 0, 0], [9, 0, 4], [0, 0, 0]])

    # Compute the divergence matrix
    div = dyn.tl.graph_calculus.divergence(adj)
    div_direct = dyn.tl.graph_calculus.divergence(adj, method="direct")
    div_weighted = dyn.tl.graph_calculus.divergence(adj, M, weighted=True)

    # Check that the matrix has the expected shape values
    assert np.all(div == div_direct)
    assert div.shape[0] == 3
    expected_data = np.array([-0.5, 1, -0.5])
    expected_data_weighted = np.array([-1.5, 2.5, -1])
    assert np.all(div == expected_data)
    assert np.all(div_weighted == expected_data_weighted)


def test_gradop():
    # Create a test adjacency matrix
    adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))

    # Compute the gradient operator
    grad = dyn.tl.graph_calculus.gradop(adj)
    grad_dense = dyn.tl.graph_calculus.gradop(adj.toarray())

    # Check that the matrix has the expected shape values
    assert np.all(grad.A == grad_dense.A)
    assert grad.shape == (4, 3)


def test_graphize_velocity_coopt(adata):
    E, _, _ = dyn.tools.graph_calculus.graphize_velocity(
        X=adata.X.A,
        V=adata.layers["velocity_S"],
    )
    assert E.shape == (adata.n_obs, adata.n_obs)

    E = dyn.tools.graph_calculus.graphize_velocity_coopt(
        X=adata.X.A,
        V=adata.layers["velocity_S"].A,
        nbrs=adata.uns["neighbors"]["indices"],
        C=adata.obsp["pearson_transition_matrix"].A,
    )
    assert E.shape == (adata.n_obs, adata.n_obs)

    E = dyn.tools.graph_calculus.graphize_velocity_coopt(
        X=adata.X.A,
        V=adata.layers["velocity_S"].A,
        nbrs=adata.uns["neighbors"]["indices"],
        U=np.random.random((adata.n_obs, adata.n_vars)),
    )
    assert E.shape == (adata.n_obs, adata.n_obs)


def test_symmetrize_discrete_vector_field():
    F = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = dyn.tools.graph_calculus.symmetrize_discrete_vector_field(F)
    assert np.all(result == np.array([[1.0, -0.5], [0.5, 4.0]]))

    result = dyn.tools.graph_calculus.symmetrize_discrete_vector_field(F, mode="sym")
    assert np.all(result == np.array([[1.0, 2.5], [2.5, 4.0]]))


def test_fp_operator():
    Q = dyn.tools.graph_calculus.fp_operator(F=np.random.rand(10, 10), D=50)
    assert Q.shape == (10, 10)


def test_triangles():
    import igraph as ig
    g = ig.Graph(edges=[(0, 1), (1, 2), (2, 0), (2, 3), (3, 0)])

    result = dyn.tools.graph_operators.triangles(g)
    assert result == [2, 1, 2, 1]

    result = dyn.tools.graph_operators._triangles(g)
    assert result == [2, 1, 2, 1]


def test_cell_and_gene_confidence(adata):
    adata.uns["pp"]["layers_norm_method"] = None
    methods = ["cosine", "consensus", "correlation", "jaccard", "hybrid"]

    for method in methods:
        dyn.tl.cell_wise_confidence(adata, method=method)
        assert method + "_velocity_confidence" in adata.obs.keys()

    dyn.tl.confident_cell_velocities(adata, group="Cell_type", lineage_dict={'Proliferating Progenitor': ['Schwann Cell']})
    assert "gene_wise_confidence" in adata.uns.keys()


def test_stationary_distribution(adata):
    adata = adata.copy()
    adata.obsp["transition_matrix"] = adata.obsp["pearson_transition_matrix"].toarray()

    dyn.tl.stationary_distribution(adata, method="other", calc_rnd=False)
    dyn.tl.stationary_distribution(adata, calc_rnd=False)

    assert "sink_steady_state_distribution" in adata.obs.keys()
    assert "source_steady_state_distribution" in adata.obs.keys()


def test_neighbors_subset():
    adata = dyn.sample_data.zebrafish()
    adata = adata[:1000, :1000].copy()
    dyn.tl.neighbors(adata)
    assert check_neighbors_completeness(adata)
    indices = np.random.randint(0, len(adata), size=100)
    _adata = adata[indices].copy()
    assert not check_neighbors_completeness(_adata)

    # check obsp keys subsetting by AnnData Obj
    neighbor_result_prefix = ""
    conn_key, dist_key, neighbor_key = generate_neighbor_keys(neighbor_result_prefix)
    check_and_recompute_neighbors(adata, result_prefix=neighbor_result_prefix)
    expected_conn_mat = adata.obsp[conn_key][indices][:, indices]
    expected_dist_mat = adata.obsp[dist_key][indices][:, indices]

    print("expected_conn_mat:", expected_conn_mat.shape)
    conn_mat = _adata.obsp[conn_key]
    dist_mat = _adata.obsp[dist_key]

    assert np.all(np.abs(expected_conn_mat - conn_mat.toarray()) < 1e-7)
    assert np.all(expected_dist_mat == dist_mat.toarray())

    # recompute and neighbor graph should be fine
    dyn.tl.neighbors(_adata)
    assert check_neighbors_completeness(_adata)


def test_broken_neighbors_check_recompute():
    adata = dyn.sample_data.zebrafish()
    adata = adata[:1000, :1000].copy()
    dyn.tl.neighbors(adata)
    assert check_neighbors_completeness(adata)
    indices = np.random.randint(0, len(adata), size=100)
    _adata = adata[indices].copy()
    assert not check_neighbors_completeness(_adata)
    check_and_recompute_neighbors(_adata)
    assert check_neighbors_completeness(_adata)


# ----------------------------------- #
# Test for utils
def smallest_distance_bf(coords):
    res = float("inf")
    for (i, c1) in enumerate(coords):
        for (j, c2) in enumerate(coords):
            if i == j:
                continue
            else:
                dist = np.linalg.norm(c1 - c2)
                res = min(res, dist)
    return res


def test_smallest_distance_simple_1():
    input_mat = np.array([[1, 2], [3, 4], [5, 6], [0, 0]])
    dist = dyn.tl.compute_smallest_distance(input_mat)
    assert abs(dist - 2.23606797749979) < 1e-7
    input_mat = np.array([[0, 0], [3, 4], [5, 6], [0, 0]])
    dist = dyn.tl.compute_smallest_distance(input_mat, use_unique_coords=False)
    assert dist == 0


def test_smallest_distance_simple_random():
    n_pts = 100
    coords = []
    for i in range(n_pts):
        coords.append(np.random.rand(2) * 1000)
    coords = np.array(coords)

    assert abs(smallest_distance_bf(coords) - dyn.tl.compute_smallest_distance(coords)) < 1e-8


def test_norm_loglikelihood():
    from scipy.stats import norm

    # Generate some data from a normal distribution
    mu = 0.0
    sigma = 2.0
    data = np.random.normal(mu, sigma, size=100)

    # Calculate the log-likelihood of the data
    ll_ground_truth = np.sum(norm.logpdf(data, mu, sigma))
    ll = dyn.tl.utils.norm_loglikelihood(data, mu, sigma)
    assert ll - ll_ground_truth < 1e-9


def test_fit_linreg():
    from dynamo.estimation.csc.utils_velocity import fit_linreg, fit_linreg_robust
    from sklearn.datasets import make_regression

    X0, y0 = make_regression(n_samples=100, n_features=1, noise=0.5, random_state=0)
    X1, y1 = make_regression(n_samples=100, n_features=1, noise=0.5, random_state=2)
    X = np.vstack([X0.T, X1.T])
    y = np.vstack([y0, y1])

    k, b, r2, all_r2 = fit_linreg(X, y, intercept=True)
    k_r, b_r, r2_r, all_r2_r = fit_linreg_robust(X, y, intercept=True)

    assert np.allclose(k, k_r, rtol=1)
    assert np.allclose(b, b_r, rtol=1)
    assert np.allclose(r2, r2_r, rtol=1)
    assert np.allclose(all_r2, all_r2_r, rtol=1)
