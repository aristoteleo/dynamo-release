import warnings

warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
import torch
from dynode.vectorfield.losses import MAD, MSE, BinomialChannel, WassersteinDistance
from dynode.vectorfield.NeuralNetModels import NNmodel_FCNN, NNmodel_Sirens
from dynode.vectorfield.samplers import (
    TimeCourseDataSampler,
    UncoupledTimeCourseDataSampler,
    VelocityDataSampler,
)
from dynode.vectorfield.utilities import mean_cosine_similarity
from dynode.vectorfield.vectorfield import Dynode
from sklearn.metrics import mean_squared_error

import dynamo as dyn
from dynamo.simulation.ODE import jacobian_bifur2genes, ode_bifur2genes, toggle
from dynamo.vectorfield.utils import (
    compute_acceleration,
    compute_curl,
    compute_curvature,
    compute_divergence,
    vector_field_function,
)

adata = dyn.sim.Simulator(motif="twogenes")
adata.obsm["X_umap"], adata.obsm["velocity_umap"] = adata.X.copy(), adata.layers["velocity"].copy()
dyn.tl.neighbors(adata, basis="umap")

adata.obs["ntr"] = 0

adata.obsm["X_pca"] = adata.obsm["X_umap"].copy()
adata.obsm["velocity_pca"] = adata.obsm["velocity_umap"].copy()

adata.var["use_for_pca"] = True
adata.var["use_for_dynamics"] = True
adata.var["use_for_transition"] = True
adata.var["gamma"] = 1
adata.var["use_for_transition"] = True

adata.layers["velocity_S"] = adata.layers["velocity"].copy()

a = np.zeros((2, 2), int)
np.fill_diagonal(a, 1)

adata.uns["PCs"] = a

adata.uns["dynamics"] = {}
adata.uns["dynamics"]["use_smoothed"] = True
adata.uns["dynamics"]["has_splicing"] = True
adata.uns["dynamics"]["has_labeling"] = False
adata.uns["dynamics"]["NTR_vel"] = False
adata.uns["dynamics"]["est_method"] = "ols"
adata.uns["dynamics"]["experiment_type"] = None

adata.obsm["X_umap"], adata.obsm["velocity_umap"] = adata.X.copy(), adata.layers["velocity"].copy()

adata.obsm["X_pca"] = adata.obsm["X_umap"].copy()
adata.obsm["velocity_pca"] = adata.obsm["velocity_umap"].copy()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Velocity_sampler = VelocityDataSampler(adata=adata, normalize_velocity=True, basis=None, device=device)

TimeCourse_sampler = None

NN = NNmodel_FCNN(original_space_size=2, latent_space_size=2)


Dynode_module = Dynode(
    NNmodel=NN,
    Velocity_sampler=Velocity_sampler,
    TimeCourse_sampler=TimeCourse_sampler,
    Velocity_ChannelModel=False,
    TimeCourse_ChannelModel=False,
    NNmodel_save_path="saved_NNmodel",
    device=device,
)

Velocity_training_params = {"NN_learning_rate": 1e-4, "x_learning_rate": 1e-3, "batch_size": 50, "loss_func": MSE()}

TimeCourse_training_params = None

Dynode_module.train(
    max_iter=1000,
    Velocity_params=Velocity_training_params,
    TimeCourse_params=TimeCourse_training_params,
)

# , a = 5, beta = 0.1, ecr = 1e-5, gamma = 0.9, lambda_ = 3, minP = 1e-5, theta = 0.75
# Dynode_module.predict_velocity = Dynode_module.predict_Velocity
dyn.vf.VectorField(adata, basis="umap", pot_curl_div=True)
dyn.vf.VectorField(adata, basis="umap", method="dynode", pot_curl_div=True, Dynode=Dynode_module)
dyn.pl.topography(adata, color="umap_ddhodge_potential")

dyn.pl.topography(
    adata,
    basis="umap",
    color="umap_ddhodge_potential",
    save_show_or_return="save",
    streamline_kwargs={
        "linewidth": None,
        "cmap": None,
        "norm": None,
        "arrowsize": 1,
        "arrowstyle": "fancy",
        "minlength": 0.1,
        "transform": None,
        "start_points": None,
        "maxlength": 4.0,
        "integration_direction": "both",
        "zorder": 3,
    },
    save_kwargs={"prefix": "./figures/dynode_topography", "ext": "png", "bbox_inches": None},
    figsize=(4, 3),
)
