import numpy as np

import dynamo


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
    dist = dynamo.tl.compute_smallest_distance(input_mat)
    assert abs(dist - 2.23606797749979) < 1e-7
    input_mat = np.array([[0, 0], [3, 4], [5, 6], [0, 0]])
    dist = dynamo.tl.compute_smallest_distance(input_mat, use_unique_coords=False)
    assert dist == 0


def test_smallest_distance_simple_random():
    n_pts = 100
    coords = []
    for i in range(n_pts):
        coords.append(np.random.rand(2) * 1000)
    coords = np.array(coords)

    assert abs(smallest_distance_bf(coords) - dynamo.tl.compute_smallest_distance(coords)) < 1e-8


if __name__ == "__main__":
    test_smallest_distance_simple_1()
    test_smallest_distance_simple_random()
