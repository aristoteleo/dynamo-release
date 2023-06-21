#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import warnings

warnings.filterwarnings('ignore')
import dynamo as dyn

filename = './data/rpe1.h5ad'

rpe1 = dyn.read(filename)

dyn.convert2float(rpe1, ['Cell_cycle_possition', 'Cell_cycle_relativePos'])

rpe1.obs.exp_type.value_counts()

rpe1[rpe1.obs.exp_type == 'Chase', :].obs.time.value_counts()

rpe1[rpe1.obs.exp_type == 'Pulse', :].obs.time.value_counts()

rpe1_kinetics = rpe1[rpe1.obs.exp_type == 'Pulse', :]
rpe1_kinetics.obs['time'] = rpe1_kinetics.obs['time'].astype(str)
rpe1_kinetics.obs.loc[rpe1_kinetics.obs['time'] == 'dmso', 'time'] = -1
rpe1_kinetics.obs['time'] = rpe1_kinetics.obs['time'].astype(float)
rpe1_kinetics = rpe1_kinetics[rpe1_kinetics.obs.time != -1, :]

rpe1_kinetics.layers['new'], rpe1_kinetics.layers['total'] = rpe1_kinetics.layers['ul'] + rpe1_kinetics.layers['sl'], \
                                                             rpe1_kinetics.layers['su'] + rpe1_kinetics.layers['sl'] + \
                                                             rpe1_kinetics.layers['uu'] + rpe1_kinetics.layers['ul']

del rpe1_kinetics.layers['uu'], rpe1_kinetics.layers['ul'], rpe1_kinetics.layers['su'], rpe1_kinetics.layers['sl']

print(rpe1_kinetics.obs.time)

rpe1_kinetics.obs.time = rpe1_kinetics.obs.time.astype('float')
rpe1_kinetics.obs.time = rpe1_kinetics.obs.time / 60  # convert minutes to hours

# # velocity
# dyn.tl.recipe_kin_data(adata=rpe1_kinetics,
#                        keep_filtered_genes=True,
#                        keep_raw_layers=True,
#                        del_2nd_moments=False,
#                        tkey='time',
#                        n_top_genes=1000,
#                        )

from dynamo.tools.dynamics import dynamics_wrapper
from dynamo.tools.dimension_reduction import reduceDimension
from dynamo.tools.cell_velocities import cell_velocities
from dynamo.preprocessing.utils import (
    del_raw_layers,
    detect_experiment_datatype,
    reset_adata_X,
)
from dynamo.preprocessing import Preprocessor

keep_filtered_cells = False
keep_filtered_genes = True
keep_raw_layers = True
del_2nd_moments = True
has_splicing, has_labeling, splicing_labeling = False, True, False
if has_splicing and has_labeling and splicing_labeling:
    layers = ["X_new", "X_total", "X_uu", "X_ul", "X_su", "X_sl"]
elif has_labeling:
    layers = ["X_new", "X_total"]

# Preprocessing
preprocessor = Preprocessor(cell_cycle_score_enable=True)
preprocessor.config_monocle_recipe(rpe1_kinetics, n_top_genes=1000)
preprocessor.size_factor_kwargs.update(
    {
        "X_total_layers": False,
        "splicing_total_layers": False,
    }
)
preprocessor.normalize_by_cells_function_kwargs.update(
    {
        "X_total_layers": False,
        "splicing_total_layers": False,
        "keep_filtered": keep_filtered_genes,
        "total_szfactor": "total_Size_Factor",
    }
)
preprocessor.filter_cells_by_outliers_kwargs["keep_filtered"] = keep_filtered_cells
preprocessor.select_genes_kwargs["keep_filtered"] = keep_filtered_genes
if True:
    reset_adata_X(rpe1_kinetics, experiment_type="kin", has_labeling=has_labeling, has_splicing=has_splicing)
preprocessor.preprocess_adata_monocle(adata=rpe1_kinetics, tkey='time', experiment_type="kin")
if not keep_raw_layers:
    del_raw_layers(rpe1_kinetics)

dynamics_wrapper(rpe1_kinetics, model="stochastic", est_method="storm-cszip", del_2nd_moments=del_2nd_moments)
reduceDimension(rpe1_kinetics, reduction_method='umap')
cell_velocities(rpe1_kinetics, basis='umap')


# dyn.tl.gene_wise_confidence(adata=rpe1_kinetics,
#                             group='cell_cycle_phase',
#                             lineage_dict={'M': 'G2-M'},
#                             ekey='M_t',
#                             vkey='velocity_T'
#                             )

rpe1_kinetics.obsm['X_RFP_GFP'] = rpe1_kinetics.obs.loc[:,
                                  ['RFP_log10_corrected', 'GFP_log10_corrected']].values.astype('float')

dyn.tl.reduceDimension(rpe1_kinetics, reduction_method='umap')
dyn.tl.cell_velocities(rpe1_kinetics, enforce=True, vkey='velocity_T', ekey='M_t', basis='RFP_GFP')
dyn.pl.streamline_plot(rpe1_kinetics, color=['cell_cycle_phase'], basis='RFP_GFP')

# # # for velocity gene-wise parameters
# # import matplotlib.pyplot as plt
# # import scanpy as sc
# # sc.set_figure_params(scanpy=True, fontsize=6)
# # plt.rcParams['font.size'] = '6'
# # dyn.configuration.set_figure_params(dpi_save=600, figsize=(17 / 3 / 2.54, 17 / 3 / 2.54 * (4 / 6)))
# #
# # save_path = './cell_wise_figures/'
# # dyn.pl.streamline_plot(rpe1_kinetics, color=['cell_cycle_phase'], vkey='velocity_T', ekey='M_t', basis='RFP_GFP',
# #                        save_show_or_return='show',
# #                        save_kwargs={"path": save_path, "prefix": 'cszip_stream_gene-wise_alpha', "dpi": 600, 'ext':'png'})
#
#
# import scvelo as scv
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.size'] = '7'
# dpi = 600
# figsize = (6, 3)
#
# well_fitted = rpe1_kinetics.var['gamma_r2'] > 0
# print(well_fitted)
# well_fitted_genes = well_fitted[well_fitted].index
# # well_fitted_genes = rpe1_kinetics.var['gamma_r2_raw'].sort_values(ascending=False).index[:1000]
# save_path = './cell_wise_figures/csp_alpha.png'
# # save_path = './cell_wise_figures/cszip_alpha_m_p_on.png'
#
# from dynamo.preprocessing.cell_cycle import get_cell_phase_genes
#
# cell_cycle_genes = get_cell_phase_genes(rpe1_kinetics, None)
# print(cell_cycle_genes)
#
# # yticklabels = [None]*len(well_fitted_genes)
#
#
# ax = scv.pl.heatmap(rpe1_kinetics,
#                     var_names=well_fitted_genes,
#                     sortby='Cell_cycle_relativePos',
#                     col_color='cell_cycle_phase',
#                     n_convolve=100,
#                     layer='cell_wise_alpha',
#                     figsize=(6, 3),
#                     show=False,
#                     colorbar=True,
#                     cbar_pos=(0.12, 0.4, 0.05, 0.18)
#                     # yticklabels=yticklabels
#                     )
# # plt.colorbar()
# # plt.savefig(save_path, dpi=dpi, figsize=figsize)
# plt.show()
#
# # # genes = ['HMGA2', 'DCBLD2', 'HIPK2']
# # dyn.configuration.set_figure_params(fontsize=6)
# # genes = ['HMGA2']
# # dyn.pl.phase_portraits(rpe1_kinetics, genes=genes, color='cell_cycle_phase', basis='RFP_GFP', vkey='velocity_T',
# #                        ekey='M_t', show_arrowed_spines=False, show_quiver=True, quiver_size=5, figsize=(6*0.53, 4*0.53))
# # genes = ['DCBLD2']
# # dyn.pl.phase_portraits(rpe1_kinetics, genes=genes, color='cell_cycle_phase', basis='RFP_GFP', vkey='velocity_T',
# #                        ekey='M_t', show_arrowed_spines=False, show_quiver=True, quiver_size=5, figsize=(6*0.53, 4*0.53))
# # genes = ['HIPK2']
# # dyn.pl.phase_portraits(rpe1_kinetics, genes=genes, color='cell_cycle_phase', basis='RFP_GFP', vkey='velocity_T',
# #                        ekey='M_t', show_arrowed_spines=False, show_quiver=True, quiver_size=5, figsize=(6*0.53, 4*0.53))
#
#
# # dyn.vf.VectorField(rpe1_kinetics, basis='RFP_GFP', map_topography=True, M=100)
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# # fig, ax = plt.subplots()
# # ax = dyn.pl.topography(rpe1_kinetics, basis='RFP_GFP', color='Cell_cycle_relativePos', ax=ax,
# #                        save_show_or_return='show', fps_basis='RFP_GFP')
#
# # # dyn.tl.cell_velocities(rpe1_kinetics, basis='pca')
# # # dyn.vf.VectorField(rpe1_kinetics, basis='pca', M=100)
# # # dyn.pp.top_pca_genes(rpe1_kinetics, n_top_genes=100)
# # # top_pca_genes = rpe1_kinetics.var.index[rpe1_kinetics.var.top_pca_genes]
# # # top_pca_genes = ["CDK4", "CDK6", "CDK2", "SKP2", "WEE1", "CDK1", "CDC20"] + list(top_pca_genes)
# #
# # dyn.tl.cell_velocities(rpe1_kinetics, basis='pca')
# # dyn.vf.VectorField(rpe1_kinetics, basis='pca', M=100)
# # # top_pca_genes = rpe1_kinetics[:, rpe1_kinetics.var['use_for_transition']].var.index.tolist()
# # # top_pca_genes = ["CDK4", "CDK6", "CDK2", "SKP2", "WEE1", "CDK1", "CDC20"] + list(top_pca_genes)
# # top_pca_genes = ["CDK4", "CDK6", "CDK2", "SKP2", "WEE1", "CDK1", "CDC20"]
# #
# # dyn.vf.jacobian(rpe1_kinetics, regulators=top_pca_genes, effectors=top_pca_genes)
# # dyn.pl.jacobian(
# #     rpe1_kinetics,
# #     regulators=top_pca_genes,
# #     effectors=top_pca_genes,
# #     basis="RFP_GFP",
# # )
# #
# # divergence_rank = dyn.vf.rank_divergence_genes(rpe1_kinetics, groups='cell_cycle_phase')
# # dyn.vf.rank_jacobian_genes(rpe1_kinetics, groups='cell_cycle_phase')
# #
# # full_reg_rank = dyn.vf.rank_jacobian_genes(rpe1_kinetics,
# #                                            groups='cell_cycle_phase',
# #                                            mode="full_reg",
# #                                            abs=True,
# #                                            output_values=True,
# #                                            return_df=True)
# # full_eff_rank = dyn.vf.rank_jacobian_genes(rpe1_kinetics,
# #                                            groups='cell_cycle_phase',
# #                                            mode='full_eff',
# #                                            abs=True,
# #                                            exclude_diagonal=True,
# #                                            output_values=True,
# #                                            return_df=True)
# # # print(full_reg_rank['G2-M'])
# # # print(full_reg_rank['G2-M'].head(2))
# #
# # # unknown_cell_type_regulators = ["E2F", "Cdk4", "Cdk6", "pRB", "pRBp", "pRBpp", "Cdk2",
# # #                                  "Skp2", "Wee1", "Cdh1", "Cdc25", "Cdk1", "Cdc20"]
# # unknown_cell_type_regulators = ["CDK4", "CDK6", "CDK2", "SKP2", "WEE1", "CDK1", "CDC20"]
# #
# # edges_list = dyn.vf.build_network_per_cluster(rpe1_kinetics,
# #                                               cluster='cell_cycle_phase',
# #                                               cluster_names=None,
# #                                               full_reg_rank=full_reg_rank,
# #                                               full_eff_rank=full_eff_rank,
# #                                               genes=np.unique(unknown_cell_type_regulators),
# #                                               n_top_genes=100)
# #
# # import networkx as nx
# #
# # print(edges_list)
# # network = nx.from_pandas_edgelist(edges_list['G1-S'], 'regulator', 'target', edge_attr='weight',
# #                                   create_using=nx.DiGraph())
# # ax = dyn.pl.arcPlot(rpe1_kinetics, cluster='cell_cycle_phase', cluster_name="G1-S", edges_list=None,
# #                     network=network, color="M_t", save_show_or_return='show')
