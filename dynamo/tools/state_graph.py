import numpy as np
from scipy.spatial.distance import cdist

def remove_redundant_points_trajectory(X, tol=1e-4, output_discard=False):
    """"""
    X = np.atleast_2d(X)
    discard = np.zeros(len(X), dtype=bool)
    if X.shape[0] > 1:
        for i in range(len(X)-1):
            dist = np.linalg.norm(X[i+1] - X[i])
            if dist < tol:
                discard[i+1] = True
        X = X[~discard]
    if output_discard:
        return X, discard
    else:
        return X

def arclength_sampling(X, step_length, t=None):
    """uniformly sample data points on an arc curve that generated from vector field predictions."""
    Y = []
    x0 = X[0]
    T = [] if t is not None else None
    t0 = t[0] if t is not None else None
    i = 1
    terminate = False
    arclength = 0
    while(i < len(X) - 1 and not terminate):
        l = 0
        for j in range(i, len(X)-1):
            tangent = X[j] - x0 if j==i else X[j] - X[j-1]
            d = np.linalg.norm(tangent)
            if l + d >= step_length:
                x = x0 if j==i else X[j-1]
                y = x + (step_length-l) * tangent/d
                if t is not None:
                    tau = t0 if j==i else t[j-1]
                    tau += (step_length-l)/d * (t[j] - tau)
                    T.append(tau)
                    t0 = tau
                Y.append(y)
                x0 = y
                i = j
                break
            else:
                l += d
        arclength += step_length
        if l + d < step_length:
            terminate = True
    if T is not None:
        return np.array(Y), arclength, T
    else:
        return np.array(Y), arclength


def classify_clone_cell_type(adata, clone, clone_column, cell_type_column, cell_type_to_excluded):
    """find the dominant cell type of all the cells that are from the same clone"""
    cell_ids = np.where(adata.obs[clone_column] == clone)[0]

    to_check = adata[cell_ids].obs[cell_type_column].value_counts().index.isin(list(cell_type_to_excluded))

    cell_type = np.where(to_check)[0]

    return cell_type

def state_graph(adata, group, basis='umap', layer=None):
    from .fate import _fate
    from .utils import fetch_states

    groups, uniq_grp = adata.obs[group], adata.obs[group].unique()
    grp_graph = np.zeros((len(uniq_grp), len(uniq_grp)))
    grp_transition = np.zeros((len(uniq_grp), len(uniq_grp)))

    for i, cur_grp in enumerate(uniq_grp):
        init_cells = adata.obs_name[groups == cur_grp]
        init_states, VecFld, t_end, valid_genes = fetch_states(adata, init_states=None, init_cells=init_cells, basis=basis,
                                                               layer=layer, average=False, t_end=None)

        t, X = _fate(VecFld, init_states, VecFld_true=None, t_end=t_end, step_size=None, direction='forward',
                              interpolation_num=250, average=False)
        len_per_cell = int(X.shape[0] / len(init_cells))

        for j in np.range(len(init_cells)):
            cur_ind = np.range(j * len_per_cell, (j + 1) * len_per_cell)
            Y, arclength, T = arclength_sampling(X[cur_ind], 0.1, t=t[cur_ind])

            for k, cur_other_grp in enumerate(set(uniq_grp).difference(cur_grp)):
                cur_other_cells = adata.obs_name[groups == cur_other_grp]
                others, _, _, _ = fetch_states(adata, init_states=None, init_cells=cur_other_cells, basis=basis, layer=layer,
                                       average=False, t_end=None)
                cd = cdist(Y, others)
                min_dists = cd.min(1)

                if np.sum(min_dists < 1e-3) > 0:
                    ind_other_cell_type = uniq_grp.index(cur_other_grp)
                    grp_graph[i, ind_other_cell_type] += 1
                    grp_transition[i, ind_other_cell_type] += np.mean(T[np.where(min_dists < 1e-3)[0]])

        grp_transition[i, :] /= grp_graph[i, :]
        grp_graph[i, :] /= len(init_cells)

    return grp_graph, grp_transition

# write function to draw the figures

if __name__ == '__main__':
    # from scipy.integrate import solve_ivp
    # def func(t, x):
    #     return -x
    #
    # def hit_ground(t, x):
    #     return x[0] - 1e-3
    #
    # hit_ground.terminal = True
    #
    # def steady_state(t, x):
    #     dxdt = func(t, x)
    #     return np.linalg.norm(np.abs(dxdt)) - 1e-5
    #
    # steady_state.terminal = True
    #
    # t = np.linspace(0, 1000, 100000)
    # ret = solve_ivp(func, [t[0], t[-1]], np.array([10, 5]), t_eval=t, events=steady_state, vectorized=True)
    #
    # t = np.linspace(0, 1e8, 100000)
    # VecFld = adata.uns['VecFld']
    # X = adata[:, adata.var.use_for_velocity].X.A
    # V_func = lambda t, x: dyn.tl.vector_field_function(x=x, t=None, VecFld=VecFld)
    # ivp_f_event = lambda t, x: np.sum(np.linalg.norm(V_func(t, x)) < 1e-5) - 1
    # ivp_f_event.terminal = True
    # ret = solve_ivp(V_func, [t[0], t[-1]], X, t_eval=t, events=ivp_f_event, vectorized=True)
    adata = dyn.read_h5ad('/Users/xqiu/Desktop/cell_tag.h5ad')
    state_graph(adata, group='Cell type annotation', basis='umap', layer=None)
