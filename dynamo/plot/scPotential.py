from ..tools.utils import update_dict
from .utils import save_fig

def show_landscape(adata,
                   Xgrid,
                   Ygrid,
                   Zgrid,
                   basis="umap",
                   save_show_or_return='show',
                   save_kwargs={},
                   ):
    """Plot the quasi-potential landscape.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains Xgrid, Ygrid and Zgrid data for visualizing potential landscape.
        Xgrid: `numpy.ndarray`
            x-coordinates of the Grid produced from the meshgrid function.
        Ygrid: `numpy.ndarray`
                y-coordinates of the Grid produced from the meshgrid function.
        Zgrid: `numpy.ndarray`
                z-coordinates or potential at each of the x/y coordinate.
        basis: `str` (default: umap)
            The method of dimension reduction. By default it is trimap. Currently it is not checked with Xgrid and Ygrid.
        save_show_or_return: {'show', 'save', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'show_landscape', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.

    Returns
    -------
        A 3D plot showing the quasi-potential of each cell state.

    """

    if "grid_Pot_" + basis in adata.uns.keys():
        Xgrid_, Ygrid_, Zgrid_ = (
            adata.uns["grid_Pot_" + basis]["Xgrid"],
            adata.uns["grid_Pot_" + basis]["Ygrid"],
            adata.uns["grid_Pot_" + basis]["Zgrid"],
        )

    Xgrid = Xgrid_ if Xgrid is None else Xgrid
    Ygrid = Ygrid_ if Ygrid is None else Ygrid
    Zgrid = Zgrid_ if Zgrid is None else Zgrid

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib.colors import LightSource

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plot the surface.
    ls = LightSource(azdeg=0, altdeg=65)
    # Shade data, creating an rgb array.
    rgb = ls.shade(Zgrid, plt.cm.RdYlBu)
    surf = ax.plot_surface(
        Xgrid,
        Ygrid,
        Zgrid,
        cmap=cm.coolwarm,
        rstride=1,
        cstride=1,
        facecolors=rgb,
        linewidth=0,
        antialiased=False,
    )
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(basis + "_1")
    ax.set_ylabel(basis + "_2")
    ax.set_zlabel("U")

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'show_landscape', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax


# show_pseudopot(Xgrid, Ygrid, Zgrid)

# % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % % -- Plot selected paths on pot. surface --
# % path_spacing = 4;
# % hold on;
# % for n_path = 1:numPaths
# %     if (    ((mod(x_path(n_path, 1), path_spacing) == 0) && (mod(y_path(n_path, 1), path_spacing) == 0)) ...
# %          || ((mod(y_path(n_path, 1), path_spacing) == 0) && (mod(x_path(n_path, 1), path_spacing) == 0)) )
# %
# % %         % *** To generate log-log surface ***
# % %         x_path(n_path, :) = x_path(n_path, :) + 0.1;
# % %         y_path(n_path, :) = y_path(n_path, :) + 0.1;
# % %         % ***
# %
# %         if (path_tag(n_path) == 1)
# %             plot3(x_path(n_path, :), y_path(n_path, :), pot_path(n_path, :) ...
# %                 , '-r' , 'MarkerSize', 1)  % plot paths
# %         elseif (path_tag(n_path) == 2)
# %             plot3(x_path(n_path, :), y_path(n_path, :), pot_path(n_path, :) ...
# %                 , '-b' , 'MarkerSize', 1)  % plot paths
# %         elseif (path_tag(n_path) == 3)
# %             plot3(x_path(n_path, :), y_path(n_path, :), pot_path(n_path, :) ...
# %                 , '-g' , 'MarkerSize', 1)  % plot paths
# %         end
# %         hold on;
# %
# %     end
# % end
