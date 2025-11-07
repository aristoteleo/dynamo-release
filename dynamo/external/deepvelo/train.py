from copy import deepcopy
from typing import Callable, Mapping
import numpy as np

import torch
from anndata import AnnData
from scvelo import logging as logg

from .trainer import Trainer
from .data_loader import data_loaders as module_data
from .model import loss as module_loss
from .model import metric as module_metric
from .model import model as module_arch
from .parse_config import ConfigParser


# a hack to make constants, see https://stackoverflow.com/questions/3203286
class MetaConstants(type):
    @property
    def default_configs(cls):
        return deepcopy(cls._default_configs)


class Constants(object, metaclass=MetaConstants):
    _default_configs = {
        "name": "DeepVelo_Base",
        "n_gpu": 1,  # whether to use GPU
        "arch": {
            "type": "VeloGCN",
            "args": {
                "layers": [64, 64],
                "dropout": 0.2,
                "fc_layer": False,
                "pred_unspliced": False,
            },
        },
        "data_loader": {
            "type": "VeloDataLoader",
            "args": {
                "shuffle": False,
                "validation_split": 0.0,
                "num_workers": 2,
                "type": "pca, t",
                "topC": 30,
                "topG": 20,
                "velocity_genes": False,
                "use_scaled_u": False,
            },
        },
        "optimizer": {
            "type": "Adam",
            "args": {"lr": 0.001, "weight_decay": 0, "amsgrad": True},
        },
        "loss": {
            "type": "mle_plus_direction",
            "args": {
                "pearson_scale": 18.0,
                "coeff_u": 1.0,
                "coeff_s": 1.0,
                "inner_batch_size": None,  # if None, will autoset the size.
                "stop_pearson_after": 1000,  # by default, set this large to avoid stopping
            },
        },
        "constraint_loss": False,
        "mask_zeros": False,
        "metrics": ["mse"],
        "lr_scheduler": {"type": "StepLR", "args": {"step_size": 1, "gamma": 0.97}},
        "trainer": {
            "epochs": 100,
            "save_dir": "saved/",
            "save_period": 1000,
            "verbosity": 1,
            "monitor": "min mse",
            "early_stop": 1000,
            "tensorboard": True,
            "grad_clip": False,
        },
    }


def train(
    adata: AnnData,
    configs: Mapping,
    verbose: bool = False,
    return_kinetic_rates: bool = True,
    callback: Callable = None,
    **kwargs,
):
    batch_size, n_genes = adata.layers["Ms"].shape
    if configs["data_loader"]["args"]["velocity_genes"]:
        n_genes = int(np.sum(adata.var["velocity_genes"]))
    configs["arch"]["args"]["n_genes"] = n_genes
    configs["data_loader"]["args"]["batch_size"] = batch_size
    config = ConfigParser(configs)
    logger = config.get_logger("train")

    # setup data_loader instances, use adata as the data_source to load inmemory data
    data_loader = config.init_obj("data_loader", module_data, data_source=adata)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    if config["arch"]["type"] in ["VeloGCN", "VeloGIN"]:
        model = config.init_obj("arch", module_arch, g=data_loader.dataset.g)
    else:
        model = config.init_obj("arch", module_arch)
    logger.info(f"Beginning training of {configs['name']} ...")
    if verbose:
        logger.info(configs)
        logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, configs["loss"]["type"])
    metrics = [getattr(module_metric, met) for met in configs["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    def callback_wrapper(epoch):
        # evaluate all and return the velocity matrix (cells, features)
        config_copy = configs["data_loader"]["args"].copy()
        config_copy.update(shuffle=False, training=False, data_source=adata)
        eval_loader = getattr(module_data, configs["data_loader"]["type"])(
            **config_copy
        )
        velo_mat, velo_mat_u, kinetic_rates = trainer.eval(
            eval_loader, return_kinetic_rates=return_kinetic_rates
        )

        if callback is not None:
            callback(adata, velo_mat, velo_mat_u, kinetic_rates, epoch)
        else:
            logg.warn(
                "Set verbose to True but no callback function provided. A possible "
                "callback function accepts at least two arguments: adata, velo_mat "
            )

    if verbose:
        trainer.train_with_epoch_callback(
            callback=callback_wrapper,
            freq=kwargs.get("freq", 30),
        )
    else:
        trainer.train()

    if configs["data_loader"]["args"]["shuffle"] == False:
        eval_loader = data_loader
    else:
        config_copy = configs["data_loader"]["args"].copy()
        config_copy.update(shuffle=False, training=False, data_source=adata)
        eval_loader = getattr(module_data, configs["data_loader"]["type"])(
            **config_copy
        )
    velo_mat, velo_mat_u, kinetic_rates = trainer.eval(
        eval_loader, return_kinetic_rates=return_kinetic_rates
    )

    print("velo_mat shape:", velo_mat.shape)
    # add velocity
    if configs["data_loader"]["args"]["velocity_genes"]:
        # the predictions only contain the velocity genes
        velocity_ = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
        idx = adata.var["velocity_genes"].values
        velocity_[:, idx] = velo_mat
        if len(velo_mat_u) > 0:
            velocity_u = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
            velocity_u[:, idx] = velo_mat_u
    else:
        velocity_ = velo_mat
        velocity_u = velo_mat_u

    assert adata.layers["Ms"].shape == velocity_.shape
    adata.layers["velocity"] = velocity_  # (cells, genes)
    if len(velo_mat_u) > 0:
        adata.layers["velocity_unspliced"] = velocity_u

    logg.hint(f"added 'velocity' (adata.layers)")
    logg.hint(f"added 'velocity_unspliced' (adata.layers)")

    if return_kinetic_rates:
        for k, v in kinetic_rates.items():
            if v is not None:
                if configs["data_loader"]["args"]["velocity_genes"]:
                    v_ = np.zeros(adata.shape, dtype=v.dtype)
                    v_[:, adata.var["velocity_genes"].values] = v
                    v = v_
                adata.layers["cell_specific_" + k] = v
                logg.hint(f"added 'cell_specific_{k}' (adata.layers)")
    return trainer
