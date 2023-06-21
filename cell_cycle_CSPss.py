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
# rpe1_kinetics = rpe1_kinetics[rpe1_kinetics.obs.time > 29, :]
# rpe1_kinetics = rpe1_kinetics[rpe1_kinetics.obs.time < 31, :]
rpe1_kinetics = rpe1_kinetics[rpe1_kinetics.obs.time < 16, :]

rpe1_kinetics.layers['new'], rpe1_kinetics.layers['total'] = rpe1_kinetics.layers['ul'] + rpe1_kinetics.layers['sl'], \
                                                             rpe1_kinetics.layers['su'] + rpe1_kinetics.layers['sl'] + \
                                                             rpe1_kinetics.layers['uu'] + rpe1_kinetics.layers['ul']

del rpe1_kinetics.layers['uu'], rpe1_kinetics.layers['ul'], rpe1_kinetics.layers['su'], rpe1_kinetics.layers['sl']

print(rpe1_kinetics.obs.time)

rpe1_kinetics.obs.time = rpe1_kinetics.obs.time.astype('float')
rpe1_kinetics.obs.time = rpe1_kinetics.obs.time / 60  # convert minutes to hours

# dyn.pp.recipe_monocle(
#     rpe1_kinetics,
#     tkey="time",
#     experiment_type="one-shot",
#     # experiment_type="kin",
#     n_top_genes=1000,
#     total_layers=False,
#     keep_raw_layers=True,
#     # feature_selection_layer="new",
# )
# dyn.tl.dynamics(rpe1_kinetics,
#                 model="deterministic",
#                 # est_method='CSP4ML_CSPss'
#                 )

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
    reset_adata_X(rpe1_kinetics, experiment_type="one-shot", has_labeling=has_labeling, has_splicing=has_splicing)
preprocessor.preprocess_adata_monocle(adata=rpe1_kinetics, tkey='time', experiment_type="one-shot")
if not keep_raw_layers:
    del_raw_layers(rpe1_kinetics)

from dynamo.tools.dynamics import dynamics_wrapper
dynamics_wrapper(
    rpe1_kinetics,
    model="stochastic",
    del_2nd_moments=del_2nd_moments,
    assumption_mRNA='ss',
    one_shot_method='storm-csp',
)
reduceDimension(rpe1_kinetics, reduction_method='umap')
cell_velocities(rpe1_kinetics, enforce=True, vkey='velocity_T', ekey='M_t', basis='umap')


rpe1_kinetics.obsm['X_RFP_GFP'] = rpe1_kinetics.obs.loc[:,
                                  ['RFP_log10_corrected', 'GFP_log10_corrected']].values.astype('float')

dyn.tl.reduceDimension(rpe1_kinetics, reduction_method='umap')
dyn.tl.cell_velocities(rpe1_kinetics, enforce=True, vkey='velocity_T', ekey='M_t', basis='RFP_GFP')
dyn.pl.streamline_plot(rpe1_kinetics, color=['cell_cycle_phase'], basis='RFP_GFP', save_show_or_return='show')


# path = './figures_new/figure4/'
# figsize = (6, 4)
# dyn.pl.streamline_plot(rpe1_kinetics, color=['cell_cycle_phase'], basis='RFP_GFP', save_show_or_return='show',
#                        save_kwargs={'prefix': 'cell_cycle_rfp_gfp_15mins_dynamo', 'ext': 'png',
#                                     "bbox_inches": None, 'dpi': 600, 'path': path}, figsize=figsize)
