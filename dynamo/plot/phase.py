import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yt
#from licpy.lic import runlic

def plotUS(adata, genes):
    u = adata[genes].layers['spliced']
    s = adata[genes].layers['unspliced']

    df = pd.DataFrame({"u": u.flatten(), "s": s.flatten(), 'gene': genes * u.shape[0]}, index = range(len(u.T.flatten())))

    # use seaborn to draw the plot:
    plt.plot()
    g = sns.relplot(x="u", y="s", col = "gene", data=df)

    for cur_axes in g.axes.flatten():
        x0, x1 = cur_axes.get_xlim()
        y0, y1 = cur_axes.get_ylim()

        points = np.linspace(min(x0, y0), max(x1, y1), 100)

        cur_axes.plot(points, points, color='red', marker=None, linestyle='--', linewidth=1.0)

def plot_fitting(adata, gene, log = True, group = False):
    """
    # add the plotting for fitting the data using only the splicing
    :param adata:
    :param gene:
    :param log:
    :param group:
    :return:
    """
    groups = [''] if group == False else np.unique(adata.obs[group])

    T = adata.obs['Time']
    gene_idx = np.where(adata.var.index.values == gene)[0][0]

    for cur_grp in groups:
        alpha, gamma, u0, l0 = adata.uns['dynamo_labeling'].loc[gene, :]
        u, l = adata.layers['U'][adata.obs[group] == cur_grp, gene_idx].toarray().squeeze(), \
                    adata.layers['L'][adata.obs[group] == cur_grp, gene_idx].toarray().squeeze()
        if log:
            u, l = np.log(u + 1), np.log(l + 1)

        t = np.linspace(T[0], T[-1], 50)

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(T, u.T, linestyle='None', marker='o', markersize=10)
        plt.plot(t, alpha/gamma + (u0 - alpha/gamma) * np.exp(-gamma*t), '--')
        plt.xlabel('time (hrs)')
        plt.title('unlabeled (' + cur_grp + ')')

        plt.subplot(122)
        plt.plot(T, l.T, linestyle='None', marker='o', markersize=10)
        plt.plot(t, l0 * np.exp(-gamma*t), '--')
        plt.xlabel('time (hrs)')
        plt.title('labeled (' + cur_grp + ')')

# moran'I on the velocity genes, etc.

# cellranger data, velocyto, comparison and phase diagram

def plot_LIC_gray(tex):
    """GET_P estimates the posterior probability and part of the energy.

    Arguments
    ---------
        Y: 'np.ndarray'
            Original data.
        V: 'np.ndarray'
            Original data.
        sigma2: 'float'
            sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        gamma: 'float'
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        a: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.

    Returns
    -------
    P: 'np.ndarray'
        Posterior probability, related to equation 27.
    E: `np.ndarray'
        Energy, related to equation 26.

    """

    tex = tex[:, ::-1]
    tex = tex.T

    M, N = tex.shape
    texture = np.empty((M, N, 4), np.float32)
    texture[:, :, 0] = tex
    texture[:, :, 1] = tex
    texture[:, :, 2] = tex
    texture[:, :, 3] = 1

#     texture = scipy.ndimage.rotate(texture,-90)
    plt.figure()
    plt.imshow(texture)


def plot_LIC(U_grid, V_grid, method = 'yt', cmap = "viridis", normalize = False, density = 1, lim=(0,1), const_alpha=False, kernellen=100, xlab='Dim 1', ylab='Dim 2', file = None):
    """Visualize vector field with quiver, streamline and line integral convolution (LIC), using velocity estimates on a grid from the associated data.
    A white noise background will be used for texture as default. Adjust the bounds of lim in the range of [0, 1] which applies
    upper and lower bounds to the values of line integral convolution and enhance the visibility of plots. When const_alpha=False,
    alpha will be weighted spatially by the values of line integral convolution; otherwise a constant value of the given alpha is used.

    Arguments
    ---------
        U_grid: 'np.ndarray'
            Original data.
        V_grid: 'np.ndarray'
            Original data.
        method: 'float'
            sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        cmap: 'float'
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        normalize: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        density: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        lim: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        const_alpha: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        kernellen: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.

    Returns
    -------
    P: 'np.ndarray'
        Posterior probability, related to equation 27.
    E: `np.ndarray'
        Energy, related to equation 26.

    """

    if method == 'yt':
        velocity_x_ori, velocity_y_ori, velocity_z_ori = U_grid, V_grid, np.zeros(U_grid.shape)
        velocity_x = np.repeat(velocity_x_ori[:, :, np.newaxis], V_grid.shape[1], axis=2)
        velocity_y = np.repeat(velocity_y_ori[:, :, np.newaxis], V_grid.shape[1], axis=2)
        velocity_z = np.repeat(velocity_z_ori[np.newaxis, :, :], V_grid.shape[1], axis=0)

        data = {}

        data["velocity_x"] = (velocity_x, "km/s")
        data["velocity_y"] = (velocity_y, "km/s")
        data["velocity_z"] = (velocity_z, "km/s")
        data["velocity_sum"] = (np.sqrt(velocity_x**2 + velocity_y**2), "km/s")

        ds = yt.load_uniform_grid(data, data["velocity_x"][0].shape, length_unit=(1.0,"Mpc"))
        slc = yt.SlicePlot(ds, "z", ["velocity_sum"])
        slc.set_cmap("velocity_sum", cmap)
        slc.set_log("velocity_sum", False)

        slc.annotate_velocity(normalize = normalize)
        slc.annotate_streamlines('velocity_x', 'velocity_y', density=density)
        slc.annotate_line_integral_convolution('velocity_x', 'velocity_y', lim=lim, const_alpha=const_alpha, kernellen=kernellen)

        slc.set_xlabel(xlab)
        slc.set_ylabel(ylab)

        slc.show()

        if file is not None:
            # plt.rc('font', family='serif', serif='Times')
            # plt.rc('text', usetex=True)
            # plt.rc('xtick', labelsize=8)
            # plt.rc('ytick', labelsize=8)
            # plt.rc('axes', labelsize=8)
            slc.save(file, mpl_kwargs = {figsize: [2, 2]})
    elif method == 'lic':
        velocyto_tex = runlic(V_grid, V_grid, 100)
        plot_LIC_gray(velocyto_tex)

