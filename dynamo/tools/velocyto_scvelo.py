"""This file provides useful functions to interact with velocyto and scvelo.

Implemented functions includes:
    Run velocyto and scvelo analysis.
    Convert adata to loom object or vice versa.
    Convert Dynamo AnnData object to scvelo AnnData object or vice versa.
"""
# from .moments import *
from typing import List, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .utils import get_vel_params
from ..dynamo_logger import main_info
from scipy.sparse import csr_matrix

from ..configuration import DKM


def vlm_to_adata(
    vlm, n_comps: int = 30, basis: str = "umap", trans_mats: Optional[dict] = None, cells_ixs: List[int] = None
) -> anndata.AnnData:
    """Conversion function from the velocyto world to the dynamo world. Code original from scSLAM-seq repository.

    Args:
        vlm (VelocytoLoom): The VelocytoLoom object that will be converted into adata.
        n_comps: The number of pc components that will be stored. Defaults to 30.
        basis: The embedding that will be used to store the vlm.ts attribute. Note that velocyto doesn't usually use
            umap as embedding although `umap` as set as default for the convenience of dynamo itself. Defaults to
            "umap".
        trans_mats: A dict of all relevant transition matrices. Defaults to None.
        cells_ixs: These are the indices of the subsampled cells. Defaults to None.

    Returns:
        The updated AnnData object.
    """

    from collections import OrderedDict

    # set obs, var
    obs, var = pd.DataFrame(vlm.ca), pd.DataFrame(vlm.ra)
    vel_params = []
    vel_params_names = []
    varm = {}
    if "CellID" in obs.keys():
        obs["obs_names"] = obs.pop("CellID")
    if "Gene" in var.keys():
        var["var_names"] = var.pop("Gene")

    if hasattr(vlm, "q"):
        vel_params_names.append("gamma_b")
        vel_params.append(vlm.q)
    if hasattr(vlm, "gammas"):
        vel_params_names.append("gamma")
        vel_params.append(vlm.gammas)
    if hasattr(vlm, "R2"):
        vel_params_names.append("gamma_r2")
        vel_params.append(vlm.R2)

    # rename clusters to louvain
    try:
        ix = np.where(obs.columns == "Clusters")[0][0]
        obs_names = list(obs.columns)
        obs_names[ix] = "louvain"
        obs.columns = obs_names

        # make louvain a categorical field
        obs["louvain"] = pd.Categorical(obs["louvain"])
    except:
        print("Could not find a filed 'Clusters' in vlm.ca.")

    # set layers basics
    layers = OrderedDict(
        unspliced=csr_matrix(vlm.U.T),
        spliced=csr_matrix(vlm.S.T),
        velocity_S=csr_matrix(vlm.velocity.T),
    )

    # set X_spliced / X_unspliced
    if hasattr(vlm, "S_norm"):
        layers["X_spliced"] = csr_matrix(2**vlm.S_norm - 1).T
    if hasattr(vlm, "U_norm"):
        layers["X_unspliced"] = csr_matrix(2**vlm.U_norm - 1).T
    if hasattr(vlm, "S_sz") and not hasattr(vlm, "S_norm"):
        layers["X_spliced"] = csr_matrix(vlm.S_sz).T
    if hasattr(vlm, "U_sz") and hasattr(vlm, "U_norm"):
        layers["X_unspliced"] = csr_matrix(vlm.U_sz).T

    # set M_s / M_u
    if hasattr(vlm, "Sx"):
        layers["M_s"] = csr_matrix(vlm.Sx).T
    if hasattr(vlm, "Ux"):
        layers["M_u"] = csr_matrix(vlm.Ux).T
    if hasattr(vlm, "Sx_sz") and not hasattr(vlm, "Sx"):
        layers["M_s"] = csr_matrix(vlm.Sx_sz).T
    if hasattr(vlm, "Ux_sz") and hasattr(vlm, "Ux"):
        layers["M_u"] = csr_matrix(vlm.Ux_sz).T

    # set obsm
    obsm = {}
    obsm[DKM.X_PCA] = vlm.pcs[:, : min(n_comps, vlm.pcs.shape[1])]
    # set basis and velocity on the basis
    if basis is not None:
        obsm["X_" + basis] = vlm.ts
        obsm["velocity_" + basis] = vlm.delta_embedding

    # set transition matrix:
    uns = {}
    if hasattr(vlm, "corrcoef"):
        uns["transition_matrix"] = vlm.corrcoef
    if hasattr(vlm, "colorandum"):
        uns["louvain_colors"] = list(np.unique(vlm.colorandum))

    # add uns annotations
    if trans_mats is not None:
        for key, value in trans_mats.items():
            uns[key] = trans_mats[key]
    if cells_ixs is not None:
        uns["cell_ixs"] = cells_ixs

    obsp = {}
    if hasattr(vlm, "embedding_knn"):
        from .connectivity import adj_to_knn

        n_neighbors = np.unique((vlm.embedding_knn > 0).sum(1)).min()
        ind_mat, dist_mat = adj_to_knn(vlm.embedding_knn, n_neighbors)
        uns["neighbors"] = {"indices": ind_mat}
        obsp = {"connectivities": vlm.embedding_knn}

    uns["pp"] = {
        "has_splicing": True,
        "has_labeling": False,
        "splicing_labeling": False,
        "has_protein": False,
        "tkey": None,
        "experiment_type": "conventional",
        "X_norm_method": None,
        "layers_norm_method": None,
    }

    uns["dynamics"] = {
        "filter_gene_mode": None,
        "t": None,
        "group": None,
        "X_data": None,
        "X_fit_data": None,
        "asspt_mRNA": "ss",
        "experiment_type": "conventional",
        "normalized": True,
        "model": "deterministic",
        "est_method": "ols",
        "has_splicing": True,
        "has_labeling": False,
        "splicing_labeling": False,
        "has_protein": False,
        "use_smoothed": True,
        "NTR_vel": False,
        "log_unnormalized": True,
        "fraction_for_deg": False,
    }

    # set X
    if hasattr(vlm, "S_norm"):
        X = csr_matrix(vlm.S_norm.T)
    else:
        X = csr_matrix(vlm.S_sz.T) if hasattr(vlm, "S_sz") else csr_matrix(vlm.S.T)

    # create an anndata object with Dynamo characteristics.
    if len(vel_params_names) > 0:
        uns["vel_params_names"] = vel_params_names
        varm["vel_params"] = vel_params

    dyn_adata = anndata.AnnData(X=X, obs=obs, obsp=obsp, obsm=obsm, var=var, varm=varm, layers=layers, uns=uns)

    return dyn_adata


def converter(
    data_in, from_type: Literal["adata", "vlm"] = "adata", to_type: Literal["adata", "vlm"] = "vlm", dir: str = "."
):
    """Convert adata to loom object or vice versa.

    Args:
        data_in (Union[vcy.VelocytoLoom, anndata.AnnData]): The object to be converted.
        from_type: The type of data_in. Defaults to "adata".
        to_type: Convert to which type. Defaults to "vlm".
        dir: The path to save the loom file. Defaults to ".".

    Raises:
        ImportError: Package velocyto not installed.

    Returns:
        the converted object.
    """

    try:
        import velocyto as vcy
    except ImportError:
        raise ImportError("You need to install the package `velocyto`." "install velocyto via `pip install velocyto`")

    if from_type == "adata":
        if to_type == "vlm":
            file = dir + "/data.loom"
            data_in.write_loom(file)
            data_out = vcy.VelocytoLoom(file)
    elif from_type == "vlm":
        if to_type == "adata":
            data_out = vlm_to_adata(data_in)

    # required by plot_phase_portraits
    data_out.ra["Gene"] = data_out.ra["var_names"]
    colors20 = np.vstack(
        (
            plt.cm.tab20b(np.linspace(0.0, 1, 20))[::2],
            plt.cm.tab20c(np.linspace(0, 1, 20))[1::2],
        )
    )

    def colormap_fun(x: np.ndarray) -> np.ndarray:
        return colors20[np.mod(x, 20)]

    data_out.colorandum = colormap_fun([1] * data_out.S.shape[1])

    return data_out


def scv_dyn_convertor(adata: anndata, mode: Literal["to_dyn", "to_scv"] = "to_dyn", kin_param_pre: str = ""):
    """Convert the adata object used in Scvelo to the adata object used by Dynamo, or vice versa.

    The use case of this method includes but not limited to:
        - Preprocess AnnData with Scvelo then estimate velocity with Dynamo.
        - Preprocess AnnData with Dynamo then estimate velocity with Scvelo.
        - Apply Dynamo analyses and visualization to the velocity estimated from Scvelo.
        - Apply Scvelo analyses and visualization to the velocity estimated from Dynamo.
    Conversion may need manual adjustments based on the use case because of the difference of velocity estimation method.
    For example, all Dynamo anndata objects will be set to the conventional experiment with the stochastic model. The
    `pp` information needs modification to enable methods not supported by Scvelo including twostep/direct kinetics,
    one-shot method, degradation method. They can be specified by setting `adata.uns["pp"]["experiment_type"]`,
    `adata.uns["pp"]["model"]`, `adata.uns["pp"]["est_method"]`.

    Args:
        adata: The adata object to be converted.
        mode: The string indicates the mode. Mode `to_dyn` will convert Scvelo anndata object to Dynamo anndata object.
            mode `to_scv` will convert Dynamo anndata to Scvelo anndata.
        kin_param_pre: The prefix to specify the velocity parameters names and values.

    Returns:
        The adata object after conversion.
    """
    main_info("Dynamo and scvelo have different preprocessing procedures and velocity estimation methods. "
              "The conversion of adata may not be optimal for every use case, requiring potential manual adjustments.")
    if mode == "to_dyn":
        main_info("Start converting Scvelo adata into Dynamo adata...")
        main_info("Scvelo data wil be converted into Dynamo adata with the conventional assumption and the"
                  "stochastic model. If this is not what you want, please change them manually.")
        if "highly_variable_genes" in adata.var.columns:
            adata.var["pass_basic_filter"] = adata.var.pop("highly_variable_genes")
            adata.var["pass_basic_filter"] = [True if item == 'True' else False for item in
                                              adata.var["pass_basic_filter"]]
        if "spliced" in adata.layers.keys():
            adata.layers["X_spliced"] = adata.layers.pop("spliced")
        if "unspliced" in adata.layers.keys():
            adata.layers["X_unspliced"] = adata.layers.pop("unspliced")
        if "highly_variable" in adata.var.columns:
            adata.var["use_for_pca"] = adata.var.pop("highly_variable")
        if "recover_dynamics" in adata.uns.keys():
            adata.uns.pop("recover_dynamics")
            adata.uns["dynamics"] = {
                "filter_gene_mode": None,
                "t": None,
                "group": None,
                "X_data": None,
                "X_fit_data": None,
                "asspt_mRNA": "ss",
                "experiment_type": "conventional",
                "normalized": True,
                "model": "stochastic",
                "est_method": "gmm",
                "has_splicing": True,
                "has_labeling": False,
                "splicing_labeling": False,
                "has_protein": False,
                "use_smoothed": True,
                "NTR_vel": False,
                "log_unnormalized": True,
                "fraction_for_deg": False,
            }
        adata.uns["pp"] = {
            "has_splicing": True,
            "has_labeling": False,
            "splicing_labeling": False,
            "has_protein": False,
            "tkey": None,
            "experiment_type": "conventional",
            "X_norm_method": None,
            "layers_norm_method": None,
        }
        if "Ms" in adata.layers.keys():
            adata.layers["M_s"] = adata.layers.pop("Ms")
        if "Mu" in adata.layers.keys():
            adata.layers["M_u"] = adata.layers.pop("Mu")
        if "velocity_u" in adata.layers.keys():
            adata.layers["velocity_U"] = adata.layers.pop("velocity_u")
        if "velocity" in adata.layers.keys():
            adata.layers["velocity_S"] = adata.layers.pop("velocity")

        vel_params = []
        vel_params_name = []

        if "fit_alpha" in adata.var.columns:
            vel_params.append(adata.var.pop("fit_alpha").values)
            vel_params_name.append("alpha")
        if "fit_beta" in adata.var.columns:
            vel_params.append(adata.var.pop("fit_beta").values)
            vel_params_name.append("beta")
        if "fit_gamma" in adata.var.columns:
            vel_params.append(adata.var.pop("fit_gamma").values)
            vel_params_name.append("gamma")
        if "fit_r2" in adata.var.columns:
            vel_params.append(adata.var.pop("fit_r2").values)
            vel_params_name.append("gamma_r2")
        if "fit_u0" in adata.var.columns:
            vel_params.append(adata.var.pop("fit_u0").values)
            vel_params_name.append("u0")
        if "fit_s0" in adata.var.columns:
            vel_params.append(adata.var.pop("fit_s0").values)
            vel_params_name.append("s0")
        if len(vel_params_name) > 0:
            adata.varm[kin_param_pre + "vel_params"] = np.array(vel_params).T
            adata.uns[kin_param_pre + "vel_params_names"] = vel_params_name
    elif mode == "to_scv":
        main_info("Start converting Dynamo adata into Scvelo adata...")
        if "pass_basic_filter" in adata.var.columns:
            adata.var["highly_variable_genes"] = adata.var.pop("pass_basic_filter")
            adata.var["highly_variable_genes"] = ["True" if item else "False" for item in
                                                  adata.var["highly_variable_genes"]]
        if "X_spliced" in adata.layers.keys():
            adata.layers["spliced"] = adata.layers.pop("X_spliced")
        if "X_unspliced" in adata.layers.keys():
            adata.layers["unspliced"] = adata.layers.pop("X_unspliced")
        if "use_for_pca" in adata.var.columns:
            adata.var["highly_variable"] = adata.var.pop("use_for_pca")
        if "pp" in adata.uns.keys():
            adata.uns.pop("pp")
        if "dynamics" in adata.uns.keys():
            adata.uns.pop("dynamics")
            adata.uns["recover_dynamics"] = {
                "fit_connected_states": True,
                "fit_basal_transcription": None,
                "use_raw": False,
            }
        if "M_s" in adata.layers.keys():
            adata.layers["Ms"] = adata.layers.pop("M_s")
        if "M_u" in adata.layers.keys():
            adata.layers["Mu"] = adata.layers.pop("M_u")
        if "velocity_U" in adata.layers.keys():
            adata.layers["velocity_u"] = adata.layers.pop("velocity_U")
        if "velocity_S" in adata.layers.keys():
            adata.layers["velocity"] = adata.layers.pop("velocity_S")

        if kin_param_pre + "vel_params" in adata.varm.keys():
            vel_params_df = get_vel_params(adata, kin_param_pre=kin_param_pre)
            if "alpha" in vel_params_df.columns:
                adata.var["fit_alpha"] = vel_params_df["alpha"]
            if "beta" in vel_params_df.columns:
                adata.var["fit_beta"] = vel_params_df["beta"]
            if "gamma" in vel_params_df.columns:
                adata.var["fit_gamma"] = vel_params_df["gamma"]
            if "gamma_r2" in vel_params_df.columns:
                adata.var["fit_r2"] = vel_params_df["gamma_r2"]
            if "u0" in vel_params_df.columns:
                adata.var["fit_u0"] = vel_params_df["u0"]
            if "s0" in vel_params_df.columns:
                adata.var["fit_s0"] = vel_params_df["s0"]
            adata.varm.pop(kin_param_pre + "vel_params")
            adata.uns.pop(kin_param_pre + "vel_params_names")
    else:
        raise NotImplementedError(f"Mode: {mode} is not implemented.")
    return adata


def run_velocyto(adata: anndata.AnnData) -> anndata.AnnData:
    """Run velocyto over the AnnData object.

    1. convert adata to vlm data
    2. set up PCA, UMAP, etc.
    3. estimate the gamma parameter

    Args:
        adata: An AnnData object.

    Returns:
        The updated AnnData object.
    """
    vlm = converter(adata)

    # U_norm: log2(U_sz + pcount)
    # vlm.U_sz: norm_factor * U
    # S_norm: log2(S_sz + pcount)
    # vlm.S_sz norm_factor * S
    # vlm.Ux: smoothed unspliced
    # vlm.Sx: smoothed spliced
    # vlm.Ux_sz: smoothed unspliced -- old code
    # vlm.Sx_sz: smoothed spliced -- old code

    vlm.normalize()  # add U_norm, U_sz, S_norm, S_sz
    vlm.perform_PCA()
    vlm.knn_imputation()  # Ux, Sx, Ux_sz, Sx_sz
    vlm.pcs = adata.X  # pcs: cell x npcs ndarray

    # vlm.Sx = vlm.S_sz
    # vlm.Ux = vlm.U_sz
    # vlm.Sx_sz = vlm.S_sz
    # vlm.Ux_sz = vlm.U_sz

    # gamma fit
    vlm.fit_gammas()  # limit_gamma = False, fit_offset = True,  use_imputed_data = False, use_size_norm = False

    # estimate velocity
    vlm.predict_U()
    vlm.calculate_velocity()

    # predict future state after dt
    vlm.calculate_shift()  # assumption = 'constant_velocity'
    vlm.extrapolate_cell_at_t()  # delta_t = 1.

    return vlm


def run_scvelo(adata: anndata.AnnData) -> anndata.AnnData:
    """Run Scvelo over the AnnData.

    1. Set up PCA, UMAP, etc.
    2. Estimate gamma and all other parameters
    3. Return results (adata.var['velocity_gamma'])

    Args:
        adata: An AnnData object.

    Raises:
        ImportError: Package scvelo not installed.

    Returns:
        The updated AnnData object.
    """

    try:
        import scvelo as scv
    except ImportError:
        raise ImportError("You need to install the package `scvelo`." "install scvelo via `pip install scvelo`")

    # scv.pp.filter_and_normalize(adata, min_counts=2, min_counts_u=1, n_top_genes=3000)
    scv.pp.moments(adata)  # , n_pcs = 12, n_neighbors = 15, mode = 'distances'
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)

    # how to fit other parameters, beta, etc.?

    return adata


def mean_var_by_time(X: np.ndarray, Time: np.ndarray) -> np.ndarray:
    """Group the data based on time and find the group's mean and var.

    Args:
        X: The data to be grouped.
        Time: The corresponding time.

    Returns:
        The mean and var of each group.
    """
    import pandas as pd

    exp_data = pd.DataFrame(X)
    exp_data["Time"] = Time

    mean_val = exp_data.groupby(["Time"]).mean()
    var_val = exp_data.groupby(["Time"]).var()

    return mean_val.values, var_val.values


# def run_dynamo_deprecated(adata, normalize=True, init_num=1, sample_method="lhs"):
#     time = adata.obs["Step"].values
#     uniqe_time = list(set(time))
#     gene_num = adata.X.shape[1]
#
#     # prepare data
#     import numpy as np
#
#     x_data = np.zeros((8, len(uniqe_time), gene_num))  # use unique time
#     uu, ul, su, sl = (
#         adata.layers["uu"].toarray(),
#         adata.layers["ul"].toarray(),
#         adata.layers["su"].toarray(),
#         adata.layers["sl"].toarray(),
#     )
#     uu = np.log2(uu + 1) if normalize else uu
#     ul = np.log2(ul + 1) if normalize else ul
#     su = np.log2(su + 1) if normalize else su
#     sl = np.log2(sl + 1) if normalize else sl
#
#     x_data[0], x_data[4] = mean_var_by_time(uu, time)
#     x_data[1], x_data[5] = mean_var_by_time(ul, time)
#     x_data[2], x_data[6] = mean_var_by_time(su, time)
#     x_data[3], x_data[7] = mean_var_by_time(sl, time)
#
#     # estimation all parameters
#     p0_range = {
#         "a": [0, 1],
#         "b": [0, 1],
#         "la": [0, 1],
#         "alpha_a": [10, 1000],
#         "alpha_i": [0, 10],
#         "sigma": [0, 1],
#         "beta": [0, 10],
#         "gamma": [0, 10],
#     }
#
#     estm = estimation(list(p0_range.values()))
#     param_out = pd.DataFrame(
#         index=adata.var.index,
#         columns=[
#             "a",
#             "b",
#             "la",
#             "alpha_a",
#             "alpha_i",
#             "sigma",
#             "beta",
#             "gamma",
#         ],
#     )
#     for i in range(gene_num):
#         cur_x_data = x_data[:, :, i].squeeze()
#         param_out.iloc[i, :], cost = estm.fit_lsq(
#             uniqe_time,
#             cur_x_data,
#             p0=None,
#             n_p0=init_num,
#             sample_method=sample_method,
#         )
#
#     # estimate only on the spliced and unspliced dataset
#
#     # estimate on the labeled and unlabeled dataset
#
#     # store the fitting result in adata.uns
#     adata.uns.update({"dynamo": param_out})
#
#     return adata
#
#
# def run_dynamo_simple_fit_deprecated(adata, log=True):
#     ncells, gene_num = adata.X.shape
#
#     # estimation all parameters
#     param_out = pd.DataFrame(index=adata.var.index, columns=["alpha", "gamma"])
#
#     u, s = adata.layers["unspliced"], adata.layers["spliced"]
#     velocity_u, velocity_s = u, s
#     for i in range(gene_num):
#         cur_u, cur_s = u[:, i], s[:, i]
#         gamma = fit_gamma(cur_u.toarray().squeeze(), cur_s.toarray().squeeze())
#         alpha = np.mean(cur_s)
#
#         velocity_u[:, i] = cur_u - cur_s * gamma
#         velocity_s[:, i] = cur_s / (1 - np.exp(-1)) - cur_u
#         param_out.iloc[i, :] = [alpha, gamma]
#
#     adata.layers["velocity_u"] = velocity_u
#     adata.layers["velocity_s"] = velocity_s
#     adata.uns.update({"dynamo_simple_fit": param_out})
#
#     return adata
#
#
# def run_dynamo_labelling_deprecated(adata, log=True, group=False):
#     ncells, gene_num = adata.X.shape
#
#     # estimation all parameters
#     T = adata.obs["Time"]
#
#     groups = [""] if group == False else np.unique(adata.obs[group])
#
#     param_out = pd.DataFrame(
#         index=adata.var.index,
#         columns=[i + "_" + j for j in groups for i in ["alpha", "gamma", "u0", "l0"]],
#     )
#     L, U = adata.layers["L"], adata.layers["U"]
#     velocity_u, velocity_s = L, U
#
#     for i in range(gene_num):
#         all_parm = []
#         for cur_grp in groups.tolist():
#             cur_L, cur_U = (
#                 (L[:, i], U[:, i])
#                 if cur_grp == ""
#                 else (
#                     L[adata.obs[group] == cur_grp, i],
#                     U[adata.obs[group] == cur_grp, i],
#                 )
#             )
#             if log:
#                 cur_U, cur_L = (
#                     np.log1p(cur_U.toarray().squeeze()),
#                     np.log1p(cur_L.toarray().squeeze()),
#                 )
#             else:
#                 cur_U, cur_L = (
#                     cur_U.toarray().squeeze(),
#                     cur_L.toarray().squeeze(),
#                 )
#
#             gamma, l0 = fit_gamma_labelling(T, cur_L, mode=None)
#             alpha, u0 = fit_alpha_labelling(T, cur_U, gamma, mode=None)
#             tmp = [alpha, gamma, u0, l0]
#             all_parm.extend(tmp)
#
#             velocity_u[:, i] = (cur_L - cur_U * gamma)[:, None]
#             velocity_s[:, i] = (cur_U / (1 - np.exp(-1)) - cur_L)[:, None]
#             adata.layers[cur_grp + "velocity_u"] = velocity_u
#             adata.layers[cur_grp + "velocity_s"] = velocity_s
#
#         param_out.iloc[i, :] = all_parm
#
#     adata.uns.update({"dynamo_labelling": param_out})
#
#     return adata
#
#
# def compare_res_deprecated(
#     adata,
#     velocyto_res,
#     svelo_res,
#     dynamo_res,
#     a_val,
#     b_val,
#     la_val,
#     alpha_a_val,
#     alpha_i_val,
#     sigma_val,
#     beta_val,
#     gamma_val,
# ):
#     """
#     function to compare results from velocyto and scvelo with our new method
#     0. retrieve gamm or gamma with other parameters from velocyto result or scvelo
#     1. plot the correlation between parameters estimated with different methods
#     2. calculate the correltion between those parameters
#     """
#     # self._offset, self._offset2, self._beta, self._gamma, self._r2, self._velocity_genes
#
#     velocyto_gammas = velocyto_res.gammas
#     scvelo_gammas = svelo_res.var["velocity_gamma"]
#
#     # scatter plot the true gammas with our result
#     plt.subplots(figsize=(15, 5))
#     plt.plot()
#     plt.subplot(131)
#     plt.plot(gamma_val, velocyto_gammas, "o")
#     plt.xlabel(r"True $\gamma$")
#     plt.ylabel(r"$\gamma$ (velocyto)")
#     plt.subplot(132)
#     plt.plot(gamma_val, scvelo_gammas, "o")
#     plt.xlabel(r"True $\gamma$")
#     plt.ylabel(r"$\gamma$ (scvelo)")
#     plt.subplot(133)
#     plt.plot(gamma_val, dynamo_res.uns["dynamo"]["gamma"], "o")
#     plt.xlabel(r"True $\gamma$")
#     plt.ylabel(r"$\gamma$ (dynamo)")
#
#     # what if we only have a small number of parameters?
#     plt.subplots(figsize=(15, 5))
#     plt.plot()
#     plt.subplot(131)
#     plt.plot(alpha_a_val, svelo_res.var["fit_alpha"], "o")
#     plt.xlabel(r"True alpha")
#     plt.ylabel(r"$\alpha$ (scvelo)")
#     plt.subplot(132)
#     plt.plot(beta_val, svelo_res.var["fit_beta"], "o")
#     plt.xlabel(r"True $\beta$")
#     plt.ylabel(r"$\beta$ (scvelo)")
#     plt.subplot(133)
#     plt.plot(gamma_val, svelo_res.var["fit_gamma"], "o")
#     plt.xlabel(r"True $\gamma$")
#     plt.ylabel(r"$\gamma$ (scvelo)")
#
#     #     param_out = pd.DataFrame(index=adata.var.index, columns=['a', 'b', 'la', 'alpha_a', 'alpha_i', 'sigma', 'beta', 'gamma'])
#     # what if we only have a small number of parameters?
#     plt.subplots(figsize=(15, 15))
#     plt.subplot(331)
#     plt.plot(a_val, adata.uns["dynamo"]["a"], "o")
#     plt.xlabel(r"True $a$")
#     plt.ylabel(r"$a$ (dynamo)")
#     plt.subplot(332)
#     plt.plot(b_val, adata.uns["dynamo"]["b"], "o")
#     plt.xlabel(r"True $b$")
#     plt.ylabel(r"$b$ (dynamo)")
#     plt.subplot(333)
#     plt.plot(la_val, adata.uns["dynamo"]["la"], "o")
#     plt.xlabel(r"True $l_a$")
#     plt.ylabel(r"$l_a$ (dynamo)")
#     plt.subplot(334)
#     plt.plot(alpha_a_val, adata.uns["dynamo"]["alpha_a"], "o")
#     plt.xlabel(r"True $\alpha_a$")
#     plt.ylabel(r"$\alpha_a$ (dynamo)")
#     plt.subplot(335)
#     plt.plot(alpha_i_val, adata.uns["dynamo"]["alpha_i"], "o")
#     plt.xlabel(r"True $\alpha_i$")
#     plt.ylabel(r"$\alpha_i$ (dynamo)")
#     plt.subplot(336)
#     plt.plot(sigma_val, adata.uns["dynamo"]["sigma"], "o")
#     plt.xlabel(r"True $\sigma$")
#     plt.ylabel(r"$\sigma$ (dynamo)")
#     plt.subplot(337)
#     plt.plot(beta_val, adata.uns["dynamo"]["beta"], "o")
#     plt.xlabel(r"True $\beta$")
#     plt.ylabel(r"$\beta$ (dynamo)")
#     plt.subplot(338)
#     plt.plot(gamma_val, adata.uns["dynamo"]["gamma"], "o")
#     plt.xlabel(r"True $\gamma$")
#     plt.ylabel(r"$\gamma$ (dynamo)")
#
#     velocyto_coef = {"gamma": np.corrcoef(gamma_val, velocyto_gammas)[1, 0]}
#     scvelo_coef = {
#         "alpha": np.corrcoef(alpha_a_val, svelo_res.var["fit_alpha"])[1, 0],
#         "beta": np.corrcoef(beta_val, svelo_res.var["fit_beta"])[1, 0],
#         "gamma": np.corrcoef(gamma_val, svelo_res.var["fit_gamma"])[1, 0],
#     }
#
#     dynamo_coef = {
#         "a": np.corrcoef(a_val, list(dynamo_res.uns["dynamo"]["a"]))[1, 0],
#         "b": np.corrcoef(b_val, list(dynamo_res.uns["dynamo"]["b"]))[1, 0],
#         "la": np.corrcoef(la_val, list(dynamo_res.uns["dynamo"]["la"]))[1, 0],
#         "alpha_a": np.corrcoef(alpha_a_val, list(dynamo_res.uns["dynamo"]["alpha_a"]))[1, 0],
#         "alpha_i": np.corrcoef(alpha_i_val, list(dynamo_res.uns["dynamo"]["alpha_i"]))[1, 0],
#         "sigma": np.corrcoef(sigma_val, list(dynamo_res.uns["dynamo"]["sigma"]))[1, 0],
#         "beta": np.corrcoef(beta_val, list(dynamo_res.uns["dynamo"]["beta"]))[1, 0],
#         "gamma": np.corrcoef(gamma_val, list(dynamo_res.uns["dynamo"]["gamma"]))[1, 0],
#     }
#
#     return {
#         "velocyto": pd.DataFrame.from_dict(velocyto_coef, orient="index").T,
#         "scvelo": pd.DataFrame.from_dict(scvelo_coef, orient="index").T,
#         "dynamo": pd.DataFrame.from_dict(dynamo_coef, orient="index").T,
#     }
