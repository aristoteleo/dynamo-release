import numpy as np
from random import uniform
from anndata import AnnData
import pandas as pd


def two_genes_motif(x,
                    a1 = 1,
                    a2 = 1,
                    b1 = 1,
                    b2 = 1,
                    k1 = 1,
                    k2 = 1,
                    S = 0.5,
                    n = 4):
    """The ODE model for the famous Pu.1-Gata.1 like network motif with self-activation and mutual inhibition. 
    """

    dx = np.nan*np.ones(x.shape)

    dx[:,0] = a1 *x[:,0]**n / (S**n +x[:,0]**n) + b1 *S**n/(S**n +x[:,1]**n) -k1*x[:,0]
    dx[:,1] = a2 *x[:,1]**n / (S**n +x[:,1]**n) + b2 *S**n/(S**n +x[:,0]**n) -k2*x[:,1]

    return dx


def neurogenesis(x,     
            mature_mu = 0,
            n = 4,
            k = 1,
            a = 4,
            eta = 0.25,
            eta_m = 0.125,
            eta_b = 0.1,
            a_s = 2.2,
            a_e = 6,
            mx = 10):
    """The ODE model for the neurogenesis system that used in benchmarking Monocle 2, Scribe and dynamo (here), original from Xiaojie Qiu, et. al, 2011.  
    """

    dx = np.nan * np.ones(shape=x.shape)

    dx[:,0] = a_s * 1 / (1 + eta**n *(x[:,4] +x[:,10] +x[:,7])**n *x[:,12]**n) - k*x[:,0]
    dx[:,1] = a * (x[:,0]**n) / (1 + x[:,0]**n + x[:,5]**n) - k*x[:,1]
    dx[:,2] = a * (x[:,1]**n) / (1 + x[:,1]**n) - k*x[:,2]
    dx[:,3] = a * (x[:,1]**n) / (1 + x[:,1]**n) - k*x[:,3]
    dx[:,4] = a_e * (x[:,2]**n + x[:,3]**n + x[:,9]**n) / (1 + x[:,2]**n + x[:,3]**n + x[:,9]**n) - k*x[:,4]
    dx[:,5] = a * (x[:,0]**n) / (1 + x[:,0]**n + x[:,1]**n) - k*x[:,5]
    dx[:,6] = a_e * (eta**n * x[:,5]**n) / (1 + eta**n * x[:,5]**n + x[:,7]**n) - k*x[:,6]
    dx[:,7] = a_e * (eta**n * x[:,5]**n) / (1 + x[:,6]**n + eta**n * x[:,5]**n) - k*x[:,7]
    dx[:,8] = a * (eta**n * x[:,5]**n * x[:,6]**n) / (1 + eta**n * x[:,5]**n * x[:,6]**n) - k*x[:,8]
    dx[:,9] = a * (x[:,7]**n) / (1 + x[:,7]**n) - k*x[:,9]
    dx[:,10] = a_e * (x[:,8]**n) / (1 + x[:,8]**n) - k*x[:,10]
    dx[:,11] = a * (eta_m**n * x[:,7]**n) / (1 + eta_m**n * x[:,7]**n) - k*x[:,11]
    dx[:,12] = mature_mu * (1 - x[:,12] / mx)

    return dx


def state_space_sampler(ode, dim, min_val=0, max_val=4, N=10000): 
    """Sample N points from the dim dimension gene expression space while restricting the values to be between min_val and max_val. Velocity vector at the sampled points will be calculated according to ode function.
    """

    X = np.array([ [uniform(min_val, max_val) for _ in range(dim)] for _ in range(N) ])
    Y = np.clip( X + ode(X), a_min=min_val, a_max=None)

    return X, Y


# Simulate the neurogenesis data via gillespie model?
def Simulator_deterministic(motif='neurogenesis'):
    """Simulate the gene expression dynamics via deterministic ODE model

    Parameters
    ----------
    motif: `str` (default: `neurogenesis`)
        Name of the network motif that will be used in the simulation.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            an Annodata object containing the simulated data.
    """

    if motif is 'neurogenesis':
        cell_num = 50000
        X, Y = state_space_sampler(ode=neurogenesis, dim=13, min_val=0, max_val=6, N=cell_num)

        gene_name = np.array(["Pax6", "Mash1", "Brn2", "Zic1", "Tuj1", "Hes5", "Scl", "Olig2", "Stat3", "Myt1L", "Alhd1L",
                              "Sox8", "Maturation"])
    elif motif is 'twogenes':
        cell_num = 5000
        X, Y = state_space_sampler(ode=two_genes_motif, dim=2, min_val=0, max_val=4, N=cell_num)
        gene_name = np.array(['Pu.1', 'Gata.1'])

    var = pd.DataFrame({'gene_short_name': gene_name})  # use the real name in simulation?
    var.set_index('gene_short_name', inplace=True)

    # provide more annotation for cells next:
    cell_ids = ['cell_%d' % (i) for i in range(cell_num)]  # first n_traj and then steps
    obs = pd.DataFrame({'Cell_name': cell_ids})
    obs.set_index('Cell_name', inplace=True)

    layers = {'velocity': Y - X}  # ambiguous is required for velocyto

    adata = AnnData(X.copy(), obs.copy(), var.copy(), layers=layers.copy())

    # remove cells that has no expression
    adata = adata[adata.X.sum(1) > 0, :]

    return adata

