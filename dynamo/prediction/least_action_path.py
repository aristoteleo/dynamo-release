import numpy as np
from scipy.optimize import minimize
import networkx as nx
from ..tools.utils import (
    nearest_neighbors,
)
from ..vectorfield.utils import (
    vecfld_from_adata,
    vector_field_function,
)
from ..vectorfield import svc_vectorfield
from .utils import remove_redundant_points_trajectory, arclength_sampling
from ..dynamo_logger import LoggerManager

def action(path, vf_func, D=1, dt=1):
    # centers
    x = (path[:-1] + path[1:]) * 0.5
    v = np.diff(path, axis=0) / dt
    
    s = (v - vf_func(x)).flatten()
    s = 0.5* s.dot(s) * dt/D
    
    return s

def action_aux(path_flatten, vf_func, dim, start=None, end=None, **kwargs):
    path = reshape_path(path_flatten, dim, start=start, end=end)
    return action(path, vf_func, **kwargs)

def action_grad(path, vf_func, jac_func, D=1, dt=1):
    x = (path[:-1] + path[1:]) * 0.5
    v = np.diff(path, axis=0) / dt
    
    dv = v - vf_func(x)
    J = jac_func(x)
    z = np.zeros(dv.shape)
    for s in range(dv.shape[0]):
        z[s] = dv[s] @ J[:,:,s]
    grad = (dv[:-1] - dv[1:])/D - dt/(2*D) * (z[:-1] + z[1:])
    return grad

def action_grad_aux(path_flatten, vf_func, jac_func, dim, start=None, end=None, **kwargs):
    path = reshape_path(path_flatten, dim, start=start, end=end)
    return action_grad(path, vf_func, jac_func, **kwargs).flatten()

def reshape_path(path_flatten, dim, start=None, end=None):
    path = path_flatten.reshape(int(len(path_flatten)/dim), dim)
    if start is not None:
        path = np.vstack((start, path))
    if end is not None:
        path = np.vstack((path, end))
    return path


def least_action_path(start, end, vf_func, jac_func, n_points=20, init_path=None, D=1, dt_0=1):
    dim = len(start)
    if init_path is None:
        path_0 = (
            np.tile(start, (n_points+1, 1)) + (np.linspace(0, 1, n_points+1, endpoint=True) * np.tile(end - start, (n_points+1, 1)).T).T
        )
    else:
        path_0 = init_path
    fun = lambda x: action_aux(x, vf_func, dim, start=path_0[0], end=path_0[-1], D=D, dt=dt_0)
    jac = lambda x: action_grad_aux(x, vf_func, jac_func, dim, start=path_0[0], end=path_0[-1], D=D, dt=dt_0)
    sol_dict = minimize(fun, path_0[1:-1], jac=jac)
    path_sol = reshape_path(sol_dict['x'], dim, start=path_0[0], end=path_0[-1])

    t_dict = minimize(lambda t: action(path_sol, vf_func, D=D, dt=t), dt_0)
    action_opt = t_dict['fun']
    dt_sol = t_dict['x'][0]
    return path_sol, dt_sol, action_opt, sol_dict


def get_init_path(G, start, end, coords, interpolation_num=20):
    source_ind = nearest_neighbors(start, coords, k=1)[0][0]
    target_ind = nearest_neighbors(end, coords, k=1)[0][0]

    path = nx.shortest_path(G, source_ind, target_ind)
    init_path = coords[path, :]

    _, arclen, _ = remove_redundant_points_trajectory(
        init_path, tol=1e-4, output_discard=True
    )
    arc_stepsize = arclen / (interpolation_num - 1)
    init_path_final, _, _ = arclength_sampling(
        init_path, step_length=arc_stepsize, t=np.arange(len(init_path))
    )

    # add the beginning and end point
    init_path_final = np.vstack((start, init_path_final, end))

    return init_path_final


def least_action(
    adata,
    start,
    end,
    basis="umap",
    vf_key='VecFld',
    vecfld=None,
    adj_key="pearson_transition_matrix",
    n_points=100,
    D=10,
):
    if vecfld is None:
        vf = svc_vectorfield()
        vf.from_adata(adata, basis=basis, vf_key=vf_key)
    else:
        vf = vecfld

    coords = adata.obsm["X_" + basis]

    # start = np.where(adata.obs.clusters == 'Ngn3 low EP')[0]
    # start = adata.obsm['X_umap'][start[0], :]

    ###

    T = adata.obsp[adj_key]
    G = nx.convert_matrix.from_scipy_sparse_matrix(T)

    # beta:
    # end = np.where(adata.obs.clusters == 'Beta')[0]
    # end = adata.obsm['X_umap'][end[-1], :]
    logger = LoggerManager.gen_logger("dynamo-least-action-path")
    logger.info("initializing path with the connectivity...", indent_level=1)
    init_path = get_init_path(G, start, end, coords, interpolation_num=n_points)

    logger.info("searching for the least action path...", indent_level=1)
    logger.log_time()
    path_sol, dt_sol, action_opt, sol_dict = least_action_path(
        start,
        end,
        vf.func,
        vf.get_Jacobian(),
        n_points=n_points,
        init_path=init_path,
        D=D,
    )
    logger.info(sol_dict['message'], indent_level=1)
    logger.finish_progress(progress_name='least action path')

    return path_sol, dt_sol, action_opt
