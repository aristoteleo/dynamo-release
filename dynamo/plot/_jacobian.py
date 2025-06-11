
import scipy as sp
import numpy as np

from ..vf import jacobian,get_jacobian
from .scatters import scatters


def jacobian_on_gene_axis(adata,
                               receptor, effector, 
                               vmin=0,
                                vmax=100,
                               x_gene=None, y_gene=None, 
                               axis_layer="M_t", 
                               temp_color_key="temp_jacobian_color", 
                               ax=None,
                               figsize=(4,4),
                               cmap="bwr",
                                sym_c=True,
                                frontier=True,
                                sort="abs",
                                alpha=0.1,
                                pointsize=0.1,
                                save_show_or_return="return",
                                despline=True,
                                deaxis=False,
                               **scatters_kwargs):
    if x_gene is None:
        x_gene = receptor
    if y_gene is None:
        y_gene = effector

    if sp.__version__<"1.14.0":
        x_axis = adata[:, x_gene].layers[axis_layer].A.flatten(),
        y_axis = adata[:, y_gene].layers[axis_layer].A.flatten(),
    else:
        x_axis = adata[:, x_gene].layers[axis_layer].toarray().flatten()
        y_axis = adata[:, y_gene].layers[axis_layer].toarray().flatten()

    jacobian(adata, regulators = [receptor, effector], effectors=[receptor, effector])
    J_df = get_jacobian(
        adata,
        receptor,
        effector,
    )
    color_values = np.full(adata.n_obs, fill_value=np.nan)
    color_values[adata.obs["pass_basic_filter"]] =  J_df.iloc[:, 0]
    adata.obs[temp_color_key] = color_values

    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)

    ax = scatters(
        adata,
        vmin=vmin,
        vmax=vmax,
        color=temp_color_key,
        cmap=cmap,
        sym_c=sym_c,
        frontier=frontier,
        sort=sort,
        alpha=alpha,
        pointsize=pointsize,
        x=x_axis,
        y=y_axis,
        save_show_or_return=save_show_or_return,
        despline=despline,
        despline_sides=["right", "top"],
        deaxis=deaxis,
        ax=ax,
        **scatters_kwargs,
    )
    ax.set_title(r"$\frac{\partial f_{%s}}{\partial x_{%s}}$" % (effector, receptor))
    ax.set_xlabel(x_gene)
    ax.set_ylabel(y_gene)
    adata.obs.pop(temp_color_key)
    return ax
