"""kdTree related wrappers around scipy and sklearn API
"""
import sklearn


def compute_smallest_distance(coords, leaf_size=40):
    if len(coords.shape) != 2:
        raise ValueError("Coordinates should be a NxM array.")
    # kd_tree = scipy.spatial.KDTree(coords)
    kd_tree = sklearn.neighbors.KDTree(coords)
    N, M = coords.shape

    # Note k=2 here because the nearest query is always a point itself.
    distances, indices = kd_tree.query(coords, k=2, return_distance=True, leaf_size=leaf_size)
    min_dist = float("inf")
    for i in range(N):
        min_dist = min(min_dist, distances[i, 1])

    return min_dist
