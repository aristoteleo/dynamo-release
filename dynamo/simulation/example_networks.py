import numpy as np
from random import uniform
from anndata import AnnData
import pandas as pd


def toggle(ab, t=None, beta=5, gamma=1, n=2):
    """Right hand side (rhs) for toggle ODEs."""
    if len(ab.shape) == 2:
        a, b = ab[:, 0], ab[:, 1]
        res = np.array([beta / (1 + b ** n) - a,
                  gamma * (beta / (1 + a ** n) - b)]).T
    else:
        a, b = ab
        res = np.array([beta / (1 + b ** n) - a,
                  gamma * (beta / (1 + a ** n) - b)])

    return res


def Ying_model(x, t=None):
    """network used in the potential landscape paper from Ying, et. al: https://www.nature.com/articles/s41598-017-15889-2"""
    if len(x.shape) == 2:
        dx1 = -1 + 9 * x[:, 0] - 2 * pow(x[:, 0], 3) + 9 * x[:, 1] - 2 * pow(x[:, 1], 3)
        dx2 = 1 - 11*x[:, 0] + 2 * pow(x[:, 0], 3) + 11 * x[:, 1] - 2 * pow(x[:, 1], 3)

        ret = np.array([dx1, dx2]).T
    else:
        dx1 = -1 + 9 * x[0] - 2 * pow(x[0], 3) + 9 * x[1] - 2 * pow(x[1], 3)
        dx2 = 1 - 11*x[0] + 2 * pow(x[0], 3) + 11 * x[1] - 2 * pow(x[1], 3)

        ret = np.array([dx1, dx2])

    return ret


def two_genes_motif(x, t=None,
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

    if len(x.shape) == 2:
        dx[:, 0] = a1 * x[:, 0] ** n / (S ** n + x[:, 0] ** n) + b1 * S ** n / (S ** n + x[:, 1] ** n) - k1 * x[:, 0]
        dx[:, 1] = a2 * x[:, 1] ** n / (S ** n + x[:, 1] ** n) + b2 * S ** n / (S ** n + x[:, 0] ** n) - k2 * x[:, 1]
    else:
        dx[0] = a1 * x[0] ** n / (S ** n + x[0] ** n) + b1 * S ** n / (S ** n + x[1] ** n) - k1 * x[0]
        dx[1] = a2 * x[1] ** n / (S ** n + x[1] ** n) + b2 * S ** n / (S ** n + x[0] ** n) - k2 * x[1]

    return dx


def neurogenesis(x, t = None,
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

    if len(x.shape) == 2:
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
    else:
        dx[0] = a_s * 1 / (1 + eta**n *(x[4] +x[10] +x[7])**n *x[12]**n) - k*x[0]
        dx[1] = a * (x[0]**n) / (1 + x[0]**n + x[5]**n) - k*x[1]
        dx[2] = a * (x[1]**n) / (1 + x[1]**n) - k*x[2]
        dx[3] = a * (x[1]**n) / (1 + x[1]**n) - k*x[3]
        dx[4] = a_e * (x[2]**n + x[3]**n + x[9]**n) / (1 + x[2]**n + x[3]**n + x[9]**n) - k*x[4]
        dx[5] = a * (x[0]**n) / (1 + x[0]**n + x[1]**n) - k*x[5]
        dx[6] = a_e * (eta**n * x[5]**n) / (1 + eta**n * x[5]**n + x[7]**n) - k*x[6]
        dx[7] = a_e * (eta**n * x[5]**n) / (1 + x[6]**n + eta**n * x[5]**n) - k*x[7]
        dx[8] = a * (eta**n * x[5]**n * x[6]**n) / (1 + eta**n * x[5]**n * x[6]**n) - k*x[8]
        dx[9] = a * (x[7]**n) / (1 + x[7]**n) - k*x[9]
        dx[10] = a_e * (x[8]**n) / (1 + x[8]**n) - k*x[10]
        dx[11] = a * (eta_m**n * x[7]**n) / (1 + eta_m**n * x[7]**n) - k*x[11]
        dx[12] = mature_mu * (1 - x[12] / mx)

    return dx


def state_space_sampler(ode, dim, clip=True, min_val=0, max_val=4, N=10000):
    """Sample N points from the dim dimension gene expression space while restricting the values to be between min_val and max_val. Velocity vector at the sampled points will be calculated according to ode function.
    """

    X = np.array([ [uniform(min_val, max_val) for _ in range(dim)] for _ in range(N) ])
    Y = np.clip( X + ode(X), a_min=min_val, a_max=None) if clip else X + ode(X)

    return X, Y


def Simulator(motif='neurogenesis', clip=True):
    """Simulate the gene expression dynamics via deterministic ODE model

    Parameters
    ----------
    motif: `str` (default: `neurogenesis`)
        Name of the network motif that will be used in the simulation.
    clip: `bool` (default: `True`)
        Whether to clip data points that are negative.  

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            an Annodata object containing the simulated data.
    """

    if motif is 'toggle':
        cell_num = 5000
        X, Y = state_space_sampler(ode=toggle, dim=2, min_val=0, max_val=6, N=cell_num)
        gene_name = np.array(['X', 'Y'])
    elif motif is 'neurogenesis':
        cell_num = 50000
        X, Y = state_space_sampler(ode=neurogenesis, dim=13, min_val=0, max_val=6, N=cell_num)

        gene_name = np.array(["Pax6", "Mash1", "Brn2", "Zic1", "Tuj1", "Hes5", "Scl", "Olig2", "Stat3", "Myt1L", "Alhd1L",
                              "Sox8", "Maturation"])
    elif motif is 'twogenes':
        cell_num = 5000
        X, Y = state_space_sampler(ode=two_genes_motif, dim=2, min_val=0, max_val=4, N=cell_num)
        gene_name = np.array(['Pu.1', 'Gata.1'])
    elif motif is 'Ying':
        cell_num = 5000
        X, Y = state_space_sampler(ode=Ying_model, dim=2, clip=clip, min_val=-3, max_val=3, N=cell_num)
        gene_name = np.array(['X', 'Y'])

    var = pd.DataFrame({'gene_short_name': gene_name})  # use the real name in simulation?
    var.set_index('gene_short_name', inplace=True)

    # provide more annotation for cells next:
    cell_ids = ['cell_%d' % (i) for i in range(cell_num)]  # first n_traj and then steps
    obs = pd.DataFrame({'Cell_name': cell_ids})
    obs.set_index('Cell_name', inplace=True)

    layers = {'velocity': Y - X}  # ambiguous is required for velocyto

    adata = AnnData(X.copy(), obs.copy(), var.copy(), layers=layers.copy())

    # remove cells that has no expression
    adata = adata[adata.X.sum(1) > 0, :] if clip else adata

    return adata


if __name__ is '__main__':
    import dynamo as dyn

    # adata_all = dyn.read_h5ad('/Volumes/xqiu/proj/Aristotle/backup/adata_all_first_10_genes.h5ad')
    # dyn.tl.dynamics(adata_all[:, :10], mode='moment', filter_gene_mode='no')

    # adata_deg = dyn.read_h5ad('/Volumes/xqiu/proj/Aristotle/backup/adata_deg_first_25_genes.h5ad')
    # dyn.tl.dynamics(adata_deg[:, :25], experiment_type='deg', filter_gene_mode='no', time_key='hour')

    # adata = dyn.read_h5ad('/Users/xqiu/Desktop/neuron_12_11.h5ad')
    # adata = dyn.read_h5ad('/Users/xqiu/Desktop/ESC_12_11.h5ad')
    adata = dyn.read_h5ad('/Users/xqiu/Desktop/neuron_12_11.h5ad')
    adata.obs['hours'] = adata.obs['time'] / 60
    # tmp = dyn.tl.dynamics(adata[:, example_genes], experiment_type='kin', filter_gene_mode='no', tkey='time')
    # dyn.pl.metabolic_labeling_fit(tmp, vkey=tmp.var_names, tkey='time', unit='minutes')
    # tmp = dyn.tl.dynamics(adata[:, example_genes], experiment_type='mix_std_stm', filter_gene_mode='no', tkey='hours')
    tmp = dyn.tl.dynamics(adata[:, example_genes], mode='moment', filter_gene_mode='no', tkey='hours')
    # dyn.pl.metabolic_labeling_fit(tmp, vkey=tmp.var_names, tkey='time', unit='minutes')
    # tmp = dyn.tl.dynamics(adata[:, example_genes], experiment_type='kin', filter_gene_mode='no', tkey='time')
    dyn.pl.dynamics(tmp, vkey=tmp.var_names, tkey='hours', unit='hours')

    # dyn.tl.dynamics(adata[:, :25], filter_gene_mode='no')
    # tmp=dyn.tl.dynamics(adata[:, :25], experiment_type='deg', filter_gene_mode='no', tkey='minutes')
    # dyn.tl.dynamics(adata[:, :25])
    # dyn.tl.dynamics(adata[:, :25], experiment_type='deg', filter_gene_mode='no', tkey='minutes')
    # dyn.tl.dynamics(adata[:, :25], experiment_type='kin', filter_gene_mode='no', tkey='minutes')
    # dyn.tl.dynamics(adata[:, :5], experiment_type='mix_std_stm', filter_gene_mode='no', tkey='minutes')
    # tmp = dyn.tl.dynamics(adata[:, :25], mode='moment', filter_gene_mode='no', tkey='minutes')
    # adata=dyn.tl.dynamics(adata[:, :25], experiment_type='deg', filter_gene_mode='no', time_key='time')
    # dyn.pl.metabolic_labeling_fit(tmp, vkey=adata.var_names[:5], tkey='minutes')
    #dyn.tl.dynamics(adata, filter_gene_mode='no', time_key='minutes')

    # toggle_adata = dyn.sim.Simulator(motif='toggle')
    # dyn.tl.VectorField(toggle_adata, basis='X', velocity_key='velocity')
    #
    # dyn.pl.topography(toggle_adata, VF=None, basis='X', init_state=None, t=np.linspace(0, 10, 200),
    #                   xlim=[0, 6], ylim=[0, 6], plot=True)

    # two_genes_adata = dyn.sim.Simulator(motif='twogenes')
    # dyn.tl.VectorField(two_genes_adata, basis='X', velocity_key='velocity')
    #
    # dyn.pl.topography(two_genes_adata, VF=None, basis='X', init_state=None, t=np.linspace(0, 10, 200),
    #                   xlim=[0, 6], ylim=[0, 6], plot=True)

    # adata=dyn.read_h5ad('/Volumes/xqiu/proj/Aristotle/backup/vector_field_hippocampus.h5ad')
    # dyn.pl.topography(adata, VF=None, basis='X', init_state=None, t=None,
    #                   xlim=[-27, 27], ylim=[-27, 27], plot=True)
    # dyn.pl.topography(two_genes_adata, VF=dyn.sim.two_genes_motif, basis='X', init_state=None, t=None, xlim=[0, 6], ylim=[0, 6], plot=True)
    # Ying_adata = dyn.sim.Simulator(motif='Ying', clip=False)
    # dyn.tl.VectorField(Ying_adata, basis='X', velocity_key='velocity', M=500, beta=2)
    #
    # dyn.pl.topography(Ying_adata, VF=None, basis='X', init_state=None, t=None,
    #                   xlim=[-3, 3], ylim=[-3, 3], plot=True, reverse=True)


