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

rpe1_genes = ['UNG', 'PCNA', 'PLK1', 'HPRT1']

rpe1_kinetics.obs.time = rpe1_kinetics.obs.time.astype('float')
rpe1_kinetics.obs.time = rpe1_kinetics.obs.time / 60  # convert minutes to hours

print(rpe1_kinetics.obs.time.value_counts())

# from dynamo.tools.recipes import recipe_kin_data
# # velocity
# recipe_kin_data(adata=rpe1_kinetics,
#                        keep_filtered_genes=True,
#                        keep_raw_layers=True,
#                        del_2nd_moments=True,
#                        tkey='time',
#                        n_top_genes=1000,
#                        # est_method='twostep',
#                        )

from dynamo.tools.dynamics import dynamics_wrapper
from dynamo.tools.dimension_reduction import reduceDimension
from dynamo.tools.cell_velocities import cell_velocities
from dynamo.preprocessing.utils import (
    del_raw_layers,
    detect_experiment_datatype,
    reset_adata_X,
    collapse_species_adata
)
from dynamo.preprocessing import Preprocessor
from dynamo.tools.moments import moments
from dynamo.preprocessing.pca import pca
from dynamo.tools.connectivity import neighbors,normalize_knn_graph
import numpy as np


keep_filtered_cells = False
keep_filtered_genes = False
keep_raw_layers = True
del_2nd_moments = True
has_splicing, has_labeling, splicing_labeling = True, True, True
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

rpe1_kinetics = collapse_species_adata(rpe1_kinetics)
if True:
    reset_adata_X(rpe1_kinetics, experiment_type="kin", has_labeling=has_labeling, has_splicing=has_splicing)
preprocessor.preprocess_adata_monocle(adata=rpe1_kinetics, tkey='time', experiment_type="kin")
if not keep_raw_layers:
    del_raw_layers(rpe1_kinetics)

tkey = rpe1_kinetics.uns["pp"]["tkey"]
# first calculate moments for labeling data relevant layers using total based connectivity graph
moments(rpe1_kinetics, group=tkey, layers=layers)

# then we want to calculate moments for spliced and unspliced layers based on connectivity graph from spliced
# data.
# first get X_spliced based pca embedding
CM = np.log1p(rpe1_kinetics[:, rpe1_kinetics.var.use_for_pca].layers["X_spliced"].A)
cm_genesums = CM.sum(axis=0)
valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
valid_ind = np.array(valid_ind).flatten()

pca(rpe1_kinetics, CM[:, valid_ind], pca_key="X_spliced_pca")
# then get neighbors graph based on X_spliced_pca
neighbors(rpe1_kinetics, X_data=rpe1_kinetics.obsm["X_spliced_pca"], layer="X_spliced")
# then normalize neighbors graph so that each row sums up to be 1
conn = normalize_knn_graph(rpe1_kinetics.obsp["connectivities"] > 0)
# then calculate moments for spliced related layers using spliced based connectivity graph
moments(rpe1_kinetics, conn=conn, layers=["X_spliced", "X_unspliced"])
# then perform kinetic estimations with properly preprocessed layers for either the labeling or the splicing
# data
moments(rpe1_kinetics, conn=conn, layers=["uu", "ul", "su", "sl", "new", "total"])

dynamics_wrapper(rpe1_kinetics, model="stochastic", est_method="storm-icsp", del_2nd_moments=del_2nd_moments)
reduceDimension(rpe1_kinetics, reduction_method='umap')
cell_velocities(rpe1_kinetics, basis='umap')

rpe1_kinetics.obsm['X_RFP_GFP'] = rpe1_kinetics.obs.loc[:,
                                  ['RFP_log10_corrected', 'GFP_log10_corrected']].values.astype('float')

# total velocity
dyn.tl.reduceDimension(rpe1_kinetics, reduction_method='umap')
dyn.tl.cell_velocities(rpe1_kinetics, enforce=True, vkey='velocity_T', ekey='M_t', basis='RFP_GFP')
dyn.pl.streamline_plot(rpe1_kinetics, color=['cell_cycle_phase'], basis='RFP_GFP')

# spliced RNA velocity
dyn.tl.reduceDimension(rpe1_kinetics, reduction_method='umap')
dyn.tl.cell_velocities(rpe1_kinetics, enforce=True, vkey='velocity_S', ekey='M_s', basis='RFP_GFP')
dyn.pl.streamline_plot(rpe1_kinetics, color=['cell_cycle_phase'], basis='RFP_GFP')

# # for velocity gene-wise parameters
# import matplotlib.pyplot as plt
# import scanpy as sc
# sc.set_figure_params(scanpy=True, fontsize=6)
# plt.rcParams['font.size'] = '6'
# dyn.configuration.set_figure_params(dpi_save=600, figsize=(17 / 3 / 2.54, 17 / 3 / 2.54 * (4 / 6)))
#
# save_path = './cell_wise_figures/'
# dyn.pl.streamline_plot(rpe1_kinetics, color=['cell_cycle_phase'], vkey='velocity_T', ekey='M_t', basis='RFP_GFP',
#                        save_show_or_return='show',
#                        save_kwargs={"path": save_path, "prefix": 'icsp_vt_stream_gene-wise_alpha_beta', "dpi": 600, 'ext':'png'})
#
#
# # spliced RNA velocity
# dyn.tl.reduceDimension(rpe1_kinetics, reduction_method='umap')
# dyn.tl.cell_velocities(rpe1_kinetics, enforce=True, vkey='velocity_S', ekey='M_s', basis='RFP_GFP')
# dyn.pl.streamline_plot(rpe1_kinetics, color=['cell_cycle_phase'], basis='RFP_GFP')
#
# # for velocity gene-wise parameters
# save_path = './cell_wise_figures/'
# dyn.pl.streamline_plot(rpe1_kinetics, color=['cell_cycle_phase'], vkey='velocity_S', ekey='M_s', basis='RFP_GFP',
#                        save_show_or_return='show',
#                        save_kwargs={"path": save_path, "prefix": 'icsp_vs_stream_gene-wise_alpha_beta', "dpi": 600, 'ext':'png'})
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
# well_fitted_genes = well_fitted[well_fitted].index
# # well_fitted_genes = rpe1_kinetics.var['gamma_r2'].sort_values(ascending=False).index[:400]
# save_path = './cell_wise_figures/icsp_beta.png'
# ax = scv.pl.heatmap(rpe1_kinetics,
#                     var_names=well_fitted_genes,
#                     sortby='Cell_cycle_relativePos',
#                     col_color='cell_cycle_phase',
#                     n_convolve=100,
#                     layer='cell_wise_beta',
#                     figsize=(6, 3),
#                     show=False)
# # plt.savefig(save_path, dpi=dpi, figsize=figsize)
# plt.show()
#
#
# # dyn.configuration.set_figure_params(fontsize=6, dpi=300)
# # genes = ['HMGA2']
# # dyn.pl.phase_portraits(rpe1_kinetics, genes=genes, color='cell_cycle_phase', basis='RFP_GFP', vkey='velocity_T',
# #                        ekey='M_t', show_arrowed_spines=False, show_quiver=True, quiver_size=5,
# #                        figsize=(6 * 0.53, 4 * 0.53))
# # genes = ['DCBLD2']
# # dyn.pl.phase_portraits(rpe1_kinetics, genes=genes, color='cell_cycle_phase', basis='RFP_GFP', vkey='velocity_T',
# #                        ekey='M_t', show_arrowed_spines=False, show_quiver=True, quiver_size=5,
# #                        figsize=(6 * 0.53, 4 * 0.53))
# # genes = ['HIPK2']
# # dyn.pl.phase_portraits(rpe1_kinetics, genes=genes, color='cell_cycle_phase', basis='RFP_GFP', vkey='velocity_T',
# #                        ekey='M_t', show_arrowed_spines=False, show_quiver=True, quiver_size=5,
# #                        figsize=(6 * 0.53, 4 * 0.53))
# #
# # # dyn.configuration.set_figure_params(fontsize=6, dpi=300)
# # # genes = ['HMGA2']
# # # dyn.pl.phase_portraits(rpe1_kinetics, genes=genes, color='cell_cycle_phase', basis='RFP_GFP', vkey='velocity_S',
# # #                        ekey='M_s', show_arrowed_spines=False, show_quiver=True, quiver_size=5,
# # #                        figsize=(6 * 0.53, 4 * 0.53))
# # # genes = ['DCBLD2']
# # # dyn.pl.phase_portraits(rpe1_kinetics, genes=genes, color='cell_cycle_phase', basis='RFP_GFP', vkey='velocity_S',
# # #                        ekey='M_s', show_arrowed_spines=False, show_quiver=True, quiver_size=5,
# # #                        figsize=(6 * 0.53, 4 * 0.53))
# # # genes = ['HIPK2']
# # # dyn.pl.phase_portraits(rpe1_kinetics, genes=genes, color='cell_cycle_phase', basis='RFP_GFP', vkey='velocity_S',
# # #                        ekey='M_s', show_arrowed_spines=False, show_quiver=True, quiver_size=5,
# # #                        figsize=(6 * 0.53, 4 * 0.53))
