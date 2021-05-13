import dynamo
import numpy as np


def test_smallest_distance_simple_1():
    input_mat = np.array([[1, 2], [3, 4], [5, 6], [0, 0]])
    dist = dynamo.tl.compute_smallest_distance(input_mat)
    print(dist)
    input_mat = np.array([[0, 0], [3, 4], [5, 6], [0, 0]])
    dist = dynamo.tl.compute_smallest_distance(input_mat)
    assert dist == 0


if __name__ == "__main__":
    test_smallest_distance_simple_1()
