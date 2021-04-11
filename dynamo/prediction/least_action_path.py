import networkx as nx
from ..tools.utils import (
    nearest_neighbors,
)
from ..vectorfield.utils import (
    vecfld_from_adata,
    vector_field_function,
)
from dynamo.vectorfield import svc_vectorfield
from ..vectorfield.least_action_path import *

from .utils import remove_redundant_points_trajectory, arclength_sampling

def get_init_path(G, start, end, coords, interpolation_num=20):
    source_ind = nearest_neighbors(start, coords, k=1)[0][0]
    target_ind = nearest_neighbors(end, coords, k=1)[0][0]

    path = nx.shortest_path(G, source_ind, target_ind)
    init_path = coords[path, :]

    _, arclen, _ = remove_redundant_points_trajectory(init_path, tol=1e-4, output_discard=True)
    arc_stepsize = arclen / (interpolation_num - 1)
    init_path_final, _, _ = arclength_sampling(init_path, step_length=arc_stepsize, t=np.arange(len(init_path)))

    # add the beginning and end point
    init_path_final = np.vstack((start, init_path_final, end))

    return init_path_final


def least_action(adata,
                 start,
                 end,
                 basis='umap',
                 adj_key='pearson_transition_matrix',
                 n_points=100,
                 D=10,
                 ):
    vecfld_dict, vecfld = vecfld_from_adata(adata, basis=basis)

    vecfld_dict_tmp = vecfld_dict.copy()
    vf = svc_vectorfield(vecfld_dict_tmp.pop('X'), vecfld_dict_tmp.pop('V'), vecfld_dict_tmp.pop('grid'),
                         **vecfld_dict_tmp)

    coords = adata.obsm['X_' + basis]

    # start = np.where(adata.obs.clusters == 'Ngn3 low EP')[0]
    # start = adata.obsm['X_umap'][start[0], :]

    ###

    T = adata.obsp[adj_key]
    G = nx.convert_matrix.from_scipy_sparse_matrix(T)

    vf.func = lambda x: vector_field_function(x, vecfld_dict)
    vf.vf_dict = vecfld_dict

    # beta:
    # end = np.where(adata.obs.clusters == 'Beta')[0]
    # end = adata.obsm['X_umap'][end[-1], :]

    init_path = get_init_path(G, start, end, coords, interpolation_num=n_points)

    path_sol, _ = least_action_path(start,
                                    end,
                                    vf.func,
                                    vf.get_Jacobian(),
                                    n_points=n_points,
                                    init_path=init_path,
                                    D=D,
                                    )
