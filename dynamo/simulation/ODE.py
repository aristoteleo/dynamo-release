from random import seed, uniform
from typing import List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd


# TODO: import from here in ..estimation.fit_jacobian.py
def hill_inh_func(x: float, A: float, K: float, n: float, g: float = 0) -> float:
    """Calculates the Hill inhibition function for a given input.

    Args:
        x: Input value for which the Hill inhibition function is to be calculated.
        A: Scaling factor for the output of the function.
        K: Concentration at which half-maximal inhibition occurs.
        n: Hill coefficient, which describes the steepness of the function's curve.
        g: Background inhibition parameter. Defaults to 0.

    Returns:
        float: The value of the Hill inhibition function for the given input.

    """
    Kd = K**n
    return A * Kd / (Kd + x**n) - g * x


def hill_inh_grad(x: float, A: float, K: float, n: float, g: float = 0) -> float:
    """Calculates the gradient of the Hill inhibition function for a given input.

    Args:
        x: Input value for which the gradient of the Hill inhibition function is to be calculated.
        A: Scaling factor for the output of the function.
        K: Concentration at which half-maximal inhibition occurs.
        n: Hill coefficient, which describes the steepness of the function's curve.
        g: Background inhibition parameter. Defaults to 0.

    Returns:
        float: The value of the gradient of the Hill inhibition function for the given input.

    """
    Kd = K**n
    return -A * n * Kd * x ** (n - 1) / (Kd + x**n) ** 2 - g


def hill_act_func(x: float, A: float, K: float, n: float, g: float = 0) -> float:
    """Calculates the Hill activation function for a given input.

    Args:
        x: Input value for which the Hill activation function is to be calculated.
        A: Scaling factor for the output of the function.
        K: Concentration at which half-maximal activation occurs.
        n: Hill coefficient, which describes the steepness of the function's curve.
        g: Background activation parameter. Defaults to 0.

    Returns:
        float: The value of the Hill activation function for the given input.

    """
    Kd = K**n
    return A * x**n / (Kd + x**n) - g * x


def hill_act_grad(x: float, A: float, K: float, n: float, g: float = 0) -> float:
    """Calculates the gradient of the Hill activation function for a given input.

    Args:
        x: Input value for which the gradient of the Hill activation function is to be calculated.
        A: Scaling factor for the output of the function.
        K: Concentration at which half-maximal activation occurs.
        n: Hill coefficient, which describes the steepness of the function's curve.
        g: Background activation parameter. Defaults to 0.

    Returns:
        float: The value of the gradient of the Hill activation function for the given input.

    """
    Kd = K**n
    return A * n * Kd * x ** (n - 1) / (Kd + x**n) ** 2 - g


def toggle(
    ab: Union[np.ndarray, Tuple[float, float]], t: Optional[float] = None, beta: float = 5, gamma: float = 1, n: int = 2
) -> np.ndarray:
    """Calculates the right-hand side (RHS) of the differential equations for the toggle switch system.

    Args:
        ab: An array or tuple containing the values of the variables a and b.
        t: Time variable. Defaults to None.
        beta: The rate of activation of a by b. Defaults to 5.
        gamma: The rate of activation of b by a. Defaults to 1.
        n: The Hill coefficient. Defaults to 2.

    Returns:
        np.ndarray: The RHS of the differential equations for the toggle switch system, calculated using the given input parameters.

    """
    if len(ab.shape) == 2:
        a, b = ab[:, 0], ab[:, 1]
        res = np.array([beta / (1 + b**n) - a, gamma * (beta / (1 + a**n) - b)]).T
    else:
        a, b = ab
        res = np.array([beta / (1 + b**n) - a, gamma * (beta / (1 + a**n) - b)])

    return res


def Ying_model(x: np.ndarray, t: Optional[float]=None):
    """network used in the potential landscape paper from Ying, et. al:
    https://www.nature.com/articles/s41598-017-15889-2.
    This is also the mixture of Gaussian model.
    """
    if len(x.shape) == 2:
        dx1 = -1 + 9 * x[:, 0] - 2 * pow(x[:, 0], 3) + 9 * x[:, 1] - 2 * pow(x[:, 1], 3)
        dx2 = 1 - 11 * x[:, 0] + 2 * pow(x[:, 0], 3) + 11 * x[:, 1] - 2 * pow(x[:, 1], 3)

        ret = np.array([dx1, dx2]).T
    else:
        dx1 = -1 + 9 * x[0] - 2 * pow(x[0], 3) + 9 * x[1] - 2 * pow(x[1], 3)
        dx2 = 1 - 11 * x[0] + 2 * pow(x[0], 3) + 11 * x[1] - 2 * pow(x[1], 3)

        ret = np.array([dx1, dx2])

    return ret


def jacobian_Ying_model(x, t=None):
    """network used in the potential landscape paper from Ying, et. al:
    https://www.nature.com/articles/s41598-017-15889-2.
    This is also the mixture of Gaussian model.
    """
    if len(x.shape) == 2:
        df1_dx1 = 9 - 6 * pow(x[:, 0], 2)
        df1_dx2 = 9 - 6 * pow(x[:, 1], 2)
        df2_dx1 = -11 + 6 * pow(x[:, 0], 2)
        df2_dx2 = 11 - 6 * pow(x[:, 1], 2)
    else:
        df1_dx1 = 9 - 6 * pow(x[0], 2)
        df1_dx2 = 9 - 6 * pow(x[1], 2)
        df2_dx1 = -11 + 6 * pow(x[0], 2)
        df2_dx2 = 11 - 6 * pow(x[1], 2)

    J = np.array([[df1_dx1, df1_dx2], [df2_dx1, df2_dx2]])

    return J


def hessian_Ying_model(x, t=None):
    """network used in the potential landscape paper from Ying, et. al:
    https://www.nature.com/articles/s41598-017-15889-2.
    This is also the mixture of Gaussian model.
    """
    if len(x.shape) == 2:
        H = np.zeros([2, 2, 2, x.shape[0]])
        H[0, 0, 0, :] = -12 * x[:, 0]
        H[1, 1, 0, :] = -12 * x[:, 1]
        H[0, 0, 1, :] = 12 * x[:, 0]
        H[1, 1, 1, :] = -12 * x[:, 1]
    else:
        H = np.zeros([2, 2, 2])
        H[0, 0, 0] = -12 * x[0]
        H[1, 1, 0] = -12 * x[1]
        H[0, 0, 1] = 12 * x[0]
        H[1, 1, 1] = -12 * x[1]

    return H


def ode_bifur2genes(
    x: np.ndarray,
    a: List = [1, 1],
    b: List = [1, 1],
    S: List = [0.5, 0.5],
    K: List = [0.5, 0.5],
    m: List = [4, 4],
    n: List = [4, 4],
    gamma: List = [1, 1],
) -> np.ndarray:
    """The ODEs for the toggle switch motif with self-activation and mutual inhibition.

    Args:
        x: The current state of the system.
        a: The self-activation strengths of the genes. Defaults to [1, 1].
        b: The mutual inhibition strengths of the genes. Defaults to [1, 1].
        S: The self-activation thresholds of the genes. Defaults to [0.5, 0.5].
        K: The mutual inhibition thresholds of the genes. Defaults to [0.5, 0.5].
        m: The Hill coefficients for self-activation. Defaults to [4, 4].
        n: The Hill coefficients for mutual inhibition. Defaults to [4, 4].
        gamma: The degradation rates of the genes. Defaults to [1, 1].

    Returns:
        np.ndarray: The rate of change of the system state.
    """

    d = x.ndim
    x = np.atleast_2d(x)
    dx = np.zeros(x.shape)

    # Compute the rate of change of each gene's concentration using Hill functions for self-activation
    # and mutual inhibition
    dx[:, 0] = hill_act_func(x[:, 0], a[0], S[0], m[0], g=gamma[0]) + hill_inh_func(x[:, 1], b[0], K[0], n[0])
    dx[:, 1] = hill_act_func(x[:, 1], a[1], S[1], m[1], g=gamma[1]) + hill_inh_func(x[:, 0], b[1], K[1], n[1])

    # Flatten the result if the input was 1-dimensional
    if d == 1:
        dx = dx.flatten()

    return dx


def jacobian_bifur2genes(
    x: np.ndarray, a=[1, 1], b=[1, 1], S=[0.5, 0.5], K=[0.5, 0.5], m=[4, 4], n=[4, 4], gamma=[1, 1]
):
    """The Jacobian of the toggle switch ODE model."""
    df1_dx1 = hill_act_grad(x[:, 0], a[0], S[0], m[0], g=gamma[0])
    df1_dx2 = hill_inh_grad(x[:, 1], b[0], K[0], n[0])
    df2_dx1 = hill_inh_grad(x[:, 0], b[1], K[1], n[1])
    df2_dx2 = hill_act_grad(x[:, 1], a[1], S[1], m[1], g=gamma[1])
    J = np.array([[df1_dx1, df1_dx2], [df2_dx1, df2_dx2]])
    return J


def two_genes_motif_jacobian(x1, x2):
    """This should be equivalent to jacobian_bifur2genes when using default parameters"""
    J = np.array(
        [
            [
                0.25 * x1**3 / (0.0625 + x1**4) ** 2 - 1,
                -0.25 * x2**3 / (0.0625 + x2**4) ** 2,
            ],
            [
                -0.25 * x1**3 / (0.0625 + x1**4) ** 2,
                0.25 * x2**3 / (0.0625 + x2**4) ** 2 - 1,
            ],
        ]
    )
    return J


def hill_inh_grad2(x, A, K, n):
    Kd = K**n
    return A * n * Kd * x ** (n - 2) * ((n + 1) * x**n - Kd * n + Kd) / (Kd + x**n) ** 3


def hill_act_grad2(x, A, K, n):
    Kd = K**n
    return -A * n * Kd * x ** (n - 2) * ((n + 1) * x**n - Kd * n + Kd) / (Kd + x**n) ** 3


def hessian_bifur2genes(x: np.ndarray, a=[1, 1], b=[1, 1], S=[0.5, 0.5], K=[0.5, 0.5], m=[4, 4], n=[4, 4], t=None):
    """The Hessian of the toggle switch ODE model."""
    if len(x.shape) == 2:
        H = np.zeros([2, 2, 2, x.shape[0]])
        H[0, 0, 0, :] = hill_act_grad2(x[:, 0], a[0], S[0], m[0])
        H[1, 1, 0, :] = hill_inh_grad2(x[:, 1], b[0], K[0], n[0])
        H[0, 0, 1, :] = hill_inh_grad2(x[:, 0], b[1], K[1], n[1])
        H[1, 1, 1, :] = hill_act_grad2(x[:, 1], a[1], S[1], m[1])
    else:
        H = np.zeros([2, 2, 2])
        H[0, 0, 0] = hill_act_grad2(x[0], a[0], S[0], m[0])
        H[1, 1, 0] = hill_inh_grad2(x[1], b[0], K[0], n[0])
        H[0, 0, 1] = hill_inh_grad2(x[0], b[1], K[1], n[1])
        H[1, 1, 1] = hill_act_grad2(x[1], a[1], S[1], m[1])

    return H


def ode_osc2genes(x: np.ndarray, a, b, S, K, m, n, gamma):
    """The ODEs for the two gene oscillation based on a predator-prey model."""

    d = x.ndim
    x = np.atleast_2d(x)
    dx = np.zeros(x.shape)

    dx[:, 0] = hill_act_func(x[:, 0], a[0], S[0], m[0], g=gamma[0]) + hill_inh_func(x[:, 1], b[0], K[0], n[0])
    dx[:, 1] = hill_act_func(x[:, 1], a[1], S[1], m[1], g=gamma[1]) + hill_act_func(x[:, 0], b[1], K[1], n[1])

    if d == 1:
        dx = dx.flatten()

    return dx


def ode_neurongenesis(
    x: np.ndarray,
    a,
    K,
    n,
    gamma,
):
    """The ODE model for the neurogenesis system that used in benchmarking Monocle 2, Scribe and dynamo (here), original from Xiaojie Qiu, et. al, 2012."""

    d = x.ndim
    x = np.atleast_2d(x)
    dx = np.zeros(shape=x.shape)

    dx[:, 0] = (
        a[0] * K[0] ** n[0] / (K[0] ** n[0] + x[:, 4] ** n[0] + x[:, 9] ** n[0] + x[:, 11] ** n[0]) - gamma[0] * x[:, 0]
    )  # Pax6
    dx[:, 1] = (
        a[1] * (x[:, 0] ** n[1]) / (K[1] ** n[1] + x[:, 0] ** n[1] + x[:, 5] ** n[1]) - gamma[1] * x[:, 1]
    )  # Mash1
    dx[:, 2] = a[2] * (x[:, 1] ** n[2]) / (K[2] ** n[2] + x[:, 1] ** n[2]) - gamma[2] * x[:, 2]  # Zic1
    dx[:, 3] = (
        a[3] * (x[:, 1] ** n[3]) / (K[3] ** n[3] + x[:, 1] ** n[3] + x[:, 7] ** n[3]) - gamma[3] * x[:, 3]
    )  # Brn2
    dx[:, 4] = (
        a[4]
        * (x[:, 2] ** n[4] + x[:, 3] ** n[4] + x[:, 10] ** n[4])
        / (K[4] ** n[4] + x[:, 2] ** n[4] + x[:, 3] ** n[4] + x[:, 10] ** n[4])  # Tuj1
        - gamma[4] * x[:, 4]
    )
    dx[:, 5] = (
        a[5] * (x[:, 0] ** n[5]) / (K[5] ** n[5] + x[:, 0] ** n[5] + x[:, 1] ** n[5]) - gamma[5] * x[:, 5]
    )  # Hes5
    dx[:, 6] = a[6] * (x[:, 5] ** n[6]) / (K[6] ** n[6] + x[:, 5] ** n[6] + x[:, 7] ** n[6]) - gamma[6] * x[:, 6]  # Scl
    dx[:, 7] = (
        a[7] * (x[:, 5] ** n[7]) / (K[7] ** n[7] + x[:, 5] ** n[7] + x[:, 6] ** n[7]) - gamma[7] * x[:, 7]
    )  # Olig2
    dx[:, 8] = (
        a[8] * (x[:, 5] ** n[8] * x[:, 6] ** n[8]) / (K[8] ** n[8] + x[:, 5] ** n[8] * x[:, 6] ** n[8])
        - gamma[8] * x[:, 8]  # Stat3
    )
    dx[:, 9] = a[9] * (x[:, 8] ** n[9]) / (K[9] ** n[9] + x[:, 8] ** n[9]) - gamma[9] * x[:, 9]  # A1dh1L
    dx[:, 10] = a[10] * (x[:, 7] ** n[10]) / (K[10] ** n[10] + x[:, 7] ** n[10]) - gamma[10] * x[:, 10]  # Myt1L
    dx[:, 11] = a[11] * (x[:, 7] ** n[11]) / (K[11] ** n[11] + x[:, 7] ** n[11]) - gamma[11] * x[:, 11]  # Sox8

    if d == 1:
        dx = dx.flatten()

    return dx


def neurongenesis(
    x,
    t=None,
    mature_mu=0,
    n=4,
    k=1,
    a=4,
    eta=0.25,
    eta_m=0.125,
    eta_b=0.1,
    a_s=2.2,
    a_e=6,
    mx=10,
):
    """The ODE model for the neurogenesis system that used in benchmarking Monocle 2, Scribe and dynamo (here), original from Xiaojie Qiu, et. al, 2012."""

    dx = np.nan * np.ones(shape=x.shape)

    if len(x.shape) == 2:
        dx[:, 0] = a_s * 1 / (1 + eta**n * (x[:, 4] + x[:, 10] + x[:, 7]) ** n * x[:, 12] ** n) - k * x[:, 0]
        dx[:, 1] = a * (x[:, 0] ** n) / (1 + x[:, 0] ** n + x[:, 5] ** n) - k * x[:, 1]
        dx[:, 2] = a * (x[:, 1] ** n) / (1 + x[:, 1] ** n) - k * x[:, 2]
        dx[:, 3] = a * (x[:, 1] ** n) / (1 + x[:, 1] ** n) - k * x[:, 3]
        dx[:, 4] = (
            a_e * (x[:, 2] ** n + x[:, 3] ** n + x[:, 9] ** n) / (1 + x[:, 2] ** n + x[:, 3] ** n + x[:, 9] ** n)
            - k * x[:, 4]
        )
        dx[:, 5] = a * (x[:, 0] ** n) / (1 + x[:, 0] ** n + x[:, 1] ** n) - k * x[:, 5]
        dx[:, 6] = a_e * (eta**n * x[:, 5] ** n) / (1 + eta**n * x[:, 5] ** n + x[:, 7] ** n) - k * x[:, 6]
        dx[:, 7] = a_e * (eta**n * x[:, 5] ** n) / (1 + x[:, 6] ** n + eta**n * x[:, 5] ** n) - k * x[:, 7]
        dx[:, 8] = (
            a * (eta**n * x[:, 5] ** n * x[:, 6] ** n) / (1 + eta**n * x[:, 5] ** n * x[:, 6] ** n) - k * x[:, 8]
        )
        dx[:, 9] = a * (x[:, 7] ** n) / (1 + x[:, 7] ** n) - k * x[:, 9]
        dx[:, 10] = a_e * (x[:, 8] ** n) / (1 + x[:, 8] ** n) - k * x[:, 10]
        dx[:, 11] = a * (eta_m**n * x[:, 7] ** n) / (1 + eta_m**n * x[:, 7] ** n) - k * x[:, 11]
        dx[:, 12] = mature_mu * (1 - x[:, 12] / mx)
    else:
        dx[0] = a_s * 1 / (1 + eta**n * (x[4] + x[10] + x[7]) ** n * x[12] ** n) - k * x[0]
        dx[1] = a * (x[0] ** n) / (1 + x[0] ** n + x[5] ** n) - k * x[1]
        dx[2] = a * (x[1] ** n) / (1 + x[1] ** n) - k * x[2]
        dx[3] = a * (x[1] ** n) / (1 + x[1] ** n) - k * x[3]
        dx[4] = a_e * (x[2] ** n + x[3] ** n + x[9] ** n) / (1 + x[2] ** n + x[3] ** n + x[9] ** n) - k * x[4]
        dx[5] = a * (x[0] ** n) / (1 + x[0] ** n + x[1] ** n) - k * x[5]
        dx[6] = a_e * (eta**n * x[5] ** n) / (1 + eta**n * x[5] ** n + x[7] ** n) - k * x[6]
        dx[7] = a_e * (eta**n * x[5] ** n) / (1 + x[6] ** n + eta**n * x[5] ** n) - k * x[7]
        dx[8] = a * (eta**n * x[5] ** n * x[6] ** n) / (1 + eta**n * x[5] ** n * x[6] ** n) - k * x[8]
        dx[9] = a * (x[7] ** n) / (1 + x[7] ** n) - k * x[9]
        dx[10] = a_e * (x[8] ** n) / (1 + x[8] ** n) - k * x[10]
        dx[11] = a * (eta_m**n * x[7] ** n) / (1 + eta_m**n * x[7] ** n) - k * x[11]
        dx[12] = mature_mu * (1 - x[12] / mx)

    return dx


def hsc():
    pass


def state_space_sampler(ode, dim, seed_num=19491001, clip=True, min_val=0, max_val=4, N=10000):
    """Sample N points from the dim dimension gene expression space while restricting the values to be between min_val and max_val. Velocity vector at the sampled points will be calculated according to ode function."""

    seed(seed)
    X = np.array([[uniform(min_val, max_val) for _ in range(dim)] for _ in range(N)])
    Y = np.clip(X + ode(X), a_min=min_val, a_max=None) if clip else X + ode(X)

    return X, Y


def Simulator(
    motif: str = "neurongenesis", seed_num=19491001, clip: Optional[bool] = None, cell_num: int = 5000
) -> anndata.AnnData:
    """Simulate the gene expression dynamics via deterministic ODE model

    Args:
        motif: str (default: "neurongenesis")
            Name of the network motif that will be used in the simulation. Can be one of {"neurongenesis", "toggle",
            "two_genes", "Ying", "mixture_of_gaussian", "four_attractors"}. The last three models are equivalent.
        clip: Whether to clip data points that are negative.
        cell_num: Number of cells to simulate.

    Returns:
        adata: an Annodata object containing the simulated data.
    """

    if motif == "toggle":
        X, Y = state_space_sampler(
            ode=toggle,
            dim=2,
            seed_num=seed_num,
            min_val=0,
            max_val=6,
            N=cell_num,
            clip=True if clip is None else clip,
        )
        gene_name = np.array(["X", "Y"])
    elif motif == "neurongenesis":
        X, Y = state_space_sampler(
            ode=neurongenesis,
            dim=13,
            seed_num=seed_num,
            min_val=0,
            max_val=6,
            N=cell_num,
            clip=True if clip is None else clip,
        )

        gene_name = np.array(
            [
                "Pax6",  #
                "Mash1",  #
                "Brn2",
                "Zic1",
                "Tuj1",
                "Hes5",  #
                "Scl",  #
                "Olig2",  #
                "Stat3",
                "Myt1L",
                "Alhd1L",
                "Sox8",
                "Maturation",
            ]
        )
    elif motif == "twogenes":
        X, Y = state_space_sampler(
            ode=ode_bifur2genes,
            dim=2,
            min_val=0,
            max_val=4,
            N=cell_num,
            clip=True if clip is None else clip,
        )
        gene_name = np.array(["Pu.1", "Gata.1"])
    elif motif in ["Ying", "mixture_of_gaussian", "four_attractors"]:
        X, Y = state_space_sampler(
            ode=Ying_model,
            dim=2,
            min_val=-3,
            max_val=3,
            N=cell_num,
            clip=False if clip is None else clip,  # Y can be smaller than 0
        )
        gene_name = np.array(["X", "Y"])

    var = pd.DataFrame({"gene_short_name": gene_name})  # use the real name in simulation?
    var.set_index("gene_short_name", inplace=True)

    # provide more annotation for cells next:
    cell_ids = ["cell_%d" % (i) for i in range(cell_num)]  # first n_traj and then steps
    obs = pd.DataFrame({"Cell_name": cell_ids})
    obs.set_index("Cell_name", inplace=True)

    layers = {"velocity": Y - X}  # ambiguous is required for velocyto

    adata = anndata.AnnData(X.copy(), obs.copy(), var.copy(), layers=layers.copy())

    # remove cells that has no expression
    adata = adata[adata.X.sum(1) > 0, :] if clip else adata

    return adata
