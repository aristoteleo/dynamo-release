import numpy as np
from sklearn.metrics import mean_squared_error

from ..tools.utils import einsum_correlation


def evaluate(reference: np.ndarray, prediction: np.ndarray, metric: str = "cosine") -> float:
    """Function to evaluate the vector field related reference quantities vs. that from vector field prediction.

    Args:
        reference: The reference quantity of the vector field (for example, simulated velocity vectors at each point or trajectory,
            or estimated RNA velocity vector)
        prediction: The predicted quantity of the vector field (for example, velocity vectors calculated based on reconstructed vector
            field function at each point or trajectory, or reconstructed RNA velocity vector)
        metric: The metric for benchmarking the vector field quantities after reconstruction.

    Returns:
        res: The score between the reference vs. reconstructed quantities based on the metric.
    """

    if metric == "cosine":
        true_normalized = reference / (np.linalg.norm(reference, axis=1).reshape(-1, 1) + 1e-20)
        predict_normalized = prediction / (np.linalg.norm(prediction, axis=1).reshape(-1, 1) + 1e-20)

        res = np.mean(true_normalized * predict_normalized) * prediction.shape[1]

    elif metric == "rmse":
        res = mean_squared_error(y_true=reference, y_pred=prediction)

    return res
