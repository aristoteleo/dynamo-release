from typing import Any, Dict, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes

from .utils import save_show_ret


def show_landscape(
    adata: AnnData,
    Xgrid: np.ndarray,
    Ygrid: np.ndarray,
    Zgrid: np.ndarray,
    basis: str = "umap",
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
) -> Optional[Axes]:
    """Plot the quasi-potential landscape.

    Args:
        adata: an AnnData object that contains Xgrid, Ygrid and Zgrid data for visualizing potential landscape.
        Xgrid: x-coordinates of the Grid produced from the meshgrid function.
        Ygrid: y-coordinates of the Grid produced from the meshgrid function.
        Zgrid: z-coordinates or potential at each of the x/y coordinate.
        basis: the method of dimension reduction. By default, it is trimap. Currently, it is not checked with Xgrid and
            Ygrid. Defaults to "umap".
        save_show_or_return: whether to save, show, or return the generated figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'show_landscape', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your need. Defaults to {}.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the matplotlib axes of the
        figure would be returned.
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

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import LightSource
    from matplotlib.ticker import FormatStrFormatter, LinearLocator
    from mpl_toolkits.mplot3d import Axes3D

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

    return save_show_ret("show_landscape", save_show_or_return, save_kwargs, ax)


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
