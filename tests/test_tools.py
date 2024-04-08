import dynamo as dyn
import numpy as np
import pandas as pd
import scipy.sparse as sp


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
    expected_data = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    expected_indices = np.array([0, 1, 0, 1, 1, 2, 1, 2])
    expected_indptr = np.array([0, 2, 4, 6, 8])
    print(grad.A)
    assert np.all(grad.data == expected_data)
    assert np.all(grad.indices == expected_indices)
    assert np.all(grad.indptr == expected_indptr)


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


if __name__ == "__main__":
    # test_calc_laplacian()
    # test_divergence()
    # test_gradop()
    # test_norm_loglikelihood()
    pass