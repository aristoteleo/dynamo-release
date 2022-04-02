from random import seed, uniform

import anndata
import numpy as np
import pandas as pd


def toggle(ab, t=None, beta=5, gamma=1, n=2):
    """Right hand side (rhs) for toggle ODEs."""
    if len(ab.shape) == 2:
        a, b = ab[:, 0], ab[:, 1]
        res = np.array([beta / (1 + b ** n) - a, gamma * (beta / (1 + a ** n) - b)]).T
    else:
        a, b = ab
        res = np.array([beta / (1 + b ** n) - a, gamma * (beta / (1 + a ** n) - b)])

    return res


def Ying_model(x, t=None):
    """network used in the potential landscape paper from Ying, et. al: https://www.nature.com/articles/s41598-017-15889-2"""
    if len(x.shape) == 2:
        dx1 = -1 + 9 * x[:, 0] - 2 * pow(x[:, 0], 3) + 9 * x[:, 1] - 2 * pow(x[:, 1], 3)
        dx2 = 1 - 11 * x[:, 0] + 2 * pow(x[:, 0], 3) + 11 * x[:, 1] - 2 * pow(x[:, 1], 3)

        ret = np.array([dx1, dx2]).T
    else:
        dx1 = -1 + 9 * x[0] - 2 * pow(x[0], 3) + 9 * x[1] - 2 * pow(x[1], 3)
        dx2 = 1 - 11 * x[0] + 2 * pow(x[0], 3) + 11 * x[1] - 2 * pow(x[1], 3)

        ret = np.array([dx1, dx2])

    return ret


def ode_2bifurgenes(x, a=[1, 1], b=[1, 1], S=[0.5, 0.5], K=[0.5, 0.5], m=[4, 4], n=[4, 4], gamma=[1, 1]):
    """The ODE model for the toggle switch motif with self-activation and mutual inhibition (e.g. Gata1-Pu.1)."""

    x = np.atleast_2d(x)
    dx = np.zeros(x.shape)

    dx[:, 0] = (
        a[0] * x[:, 0] ** m[0] / (S[0] ** m[0] + x[:, 0] ** m[0])
        + b[0] * K[0] ** n[0] / (K[0] ** n[0] + x[:, 1] ** n[0])
        - gamma[0] * x[:, 0]
    )
    dx[:, 1] = (
        a[1] * x[:, 1] ** m[1] / (S[1] ** m[1] + x[:, 1] ** m[1])
        + b[1] * K[1] ** n[1] / (K[1] ** n[1] + x[:, 0] ** n[1])
        - gamma[1] * x[:, 1]
    )

    return dx


def jacobian_2bifurgenes(x, a=[1, 1], b=[1, 1], S=[0.5, 0.5], K=[0.5, 0.5], m=[4, 4], n=[4, 4], gamma=[1, 1]):
    """The Jacobian of the toggle switch ODE model."""
    df1_dx1 = a[0] * m[0] * S[0] ** m[0] * x[:, 0] ** (m[0] - 1) / (S[0] ** m[0] + x[:, 0] ** m[0]) ** 2 - gamma[0]
    df1_dx2 = -b[0] * n[0] * K[0] ** n[0] * x[:, 1] ** (n[0] - 1) / (K[0] ** n[0] + x[:, 1] ** n[0]) ** 2
    df2_dx1 = a[1] * m[1] * S[1] ** m[1] * x[:, 1] ** (m[1] - 1) / (S[1] ** m[1] + x[:, 1] ** m[1]) ** 2 - gamma[1]
    df2_dx2 = -b[1] * n[1] * K[1] ** n[1] * x[:, 0] ** (n[1] - 1) / (K[1] ** n[1] + x[:, 0] ** n[1]) ** 2
    J = np.array([[df1_dx1, df1_dx2], [df2_dx1, df2_dx2]])
    return J


def neurogenesis(
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
    """The ODE model for the neurogenesis system that used in benchmarking Monocle 2, Scribe and dynamo (here), original from Xiaojie Qiu, et. al, 2011."""

    dx = np.nan * np.ones(shape=x.shape)

    if len(x.shape) == 2:
        dx[:, 0] = a_s * 1 / (1 + eta ** n * (x[:, 4] + x[:, 10] + x[:, 7]) ** n * x[:, 12] ** n) - k * x[:, 0]
        dx[:, 1] = a * (x[:, 0] ** n) / (1 + x[:, 0] ** n + x[:, 5] ** n) - k * x[:, 1]
        dx[:, 2] = a * (x[:, 1] ** n) / (1 + x[:, 1] ** n) - k * x[:, 2]
        dx[:, 3] = a * (x[:, 1] ** n) / (1 + x[:, 1] ** n) - k * x[:, 3]
        dx[:, 4] = (
            a_e * (x[:, 2] ** n + x[:, 3] ** n + x[:, 9] ** n) / (1 + x[:, 2] ** n + x[:, 3] ** n + x[:, 9] ** n)
            - k * x[:, 4]
        )
        dx[:, 5] = a * (x[:, 0] ** n) / (1 + x[:, 0] ** n + x[:, 1] ** n) - k * x[:, 5]
        dx[:, 6] = a_e * (eta ** n * x[:, 5] ** n) / (1 + eta ** n * x[:, 5] ** n + x[:, 7] ** n) - k * x[:, 6]
        dx[:, 7] = a_e * (eta ** n * x[:, 5] ** n) / (1 + x[:, 6] ** n + eta ** n * x[:, 5] ** n) - k * x[:, 7]
        dx[:, 8] = (
            a * (eta ** n * x[:, 5] ** n * x[:, 6] ** n) / (1 + eta ** n * x[:, 5] ** n * x[:, 6] ** n) - k * x[:, 8]
        )
        dx[:, 9] = a * (x[:, 7] ** n) / (1 + x[:, 7] ** n) - k * x[:, 9]
        dx[:, 10] = a_e * (x[:, 8] ** n) / (1 + x[:, 8] ** n) - k * x[:, 10]
        dx[:, 11] = a * (eta_m ** n * x[:, 7] ** n) / (1 + eta_m ** n * x[:, 7] ** n) - k * x[:, 11]
        dx[:, 12] = mature_mu * (1 - x[:, 12] / mx)
    else:
        dx[0] = a_s * 1 / (1 + eta ** n * (x[4] + x[10] + x[7]) ** n * x[12] ** n) - k * x[0]
        dx[1] = a * (x[0] ** n) / (1 + x[0] ** n + x[5] ** n) - k * x[1]
        dx[2] = a * (x[1] ** n) / (1 + x[1] ** n) - k * x[2]
        dx[3] = a * (x[1] ** n) / (1 + x[1] ** n) - k * x[3]
        dx[4] = a_e * (x[2] ** n + x[3] ** n + x[9] ** n) / (1 + x[2] ** n + x[3] ** n + x[9] ** n) - k * x[4]
        dx[5] = a * (x[0] ** n) / (1 + x[0] ** n + x[1] ** n) - k * x[5]
        dx[6] = a_e * (eta ** n * x[5] ** n) / (1 + eta ** n * x[5] ** n + x[7] ** n) - k * x[6]
        dx[7] = a_e * (eta ** n * x[5] ** n) / (1 + x[6] ** n + eta ** n * x[5] ** n) - k * x[7]
        dx[8] = a * (eta ** n * x[5] ** n * x[6] ** n) / (1 + eta ** n * x[5] ** n * x[6] ** n) - k * x[8]
        dx[9] = a * (x[7] ** n) / (1 + x[7] ** n) - k * x[9]
        dx[10] = a_e * (x[8] ** n) / (1 + x[8] ** n) - k * x[10]
        dx[11] = a * (eta_m ** n * x[7] ** n) / (1 + eta_m ** n * x[7] ** n) - k * x[11]
        dx[12] = mature_mu * (1 - x[12] / mx)

    return dx


def state_space_sampler(ode, dim, seed_num=19491001, clip=True, min_val=0, max_val=4, N=10000):
    """Sample N points from the dim dimension gene expression space while restricting the values to be between min_val and max_val. Velocity vector at the sampled points will be calculated according to ode function."""

    seed(seed)
    X = np.array([[uniform(min_val, max_val) for _ in range(dim)] for _ in range(N)])
    Y = np.clip(X + ode(X), a_min=min_val, a_max=None) if clip else X + ode(X)

    return X, Y


def Simulator(motif="neurogenesis", seed_num=19491001, clip=True, cell_num=5000):
    """Simulate the gene expression dynamics via deterministic ODE model

    Parameters
    ----------
    motif: str (default: "neurogenesis")
        Name of the network motif that will be used in the simulation.
    clip: bool (default: True)
        Whether to clip data points that are negative.
    cell_num: int (default: 5000)
        Number of cells to simulate.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            an Annodata object containing the simulated data.
    """

    if motif == "toggle":
        X, Y = state_space_sampler(
            ode=toggle,
            dim=2,
            seed_num=seed_num,
            min_val=0,
            max_val=6,
            N=cell_num,
        )
        gene_name = np.array(["X", "Y"])
    elif motif == "neurogenesis":
        X, Y = state_space_sampler(
            ode=neurogenesis,
            dim=13,
            seed_num=seed_num,
            min_val=0,
            max_val=6,
            N=cell_num,
        )

        gene_name = np.array(
            [
                "Pax6",
                "Mash1",
                "Brn2",
                "Zic1",
                "Tuj1",
                "Hes5",
                "Scl",
                "Olig2",
                "Stat3",
                "Myt1L",
                "Alhd1L",
                "Sox8",
                "Maturation",
            ]
        )
    elif motif == "twogenes":
        X, Y = state_space_sampler(ode=ode_2bifurgenes, dim=2, min_val=0, max_val=4, N=cell_num)
        gene_name = np.array(["Pu.1", "Gata.1"])
    elif motif == "Ying":
        X, Y = state_space_sampler(ode=Ying_model, dim=2, clip=clip, min_val=-3, max_val=3, N=cell_num)
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
