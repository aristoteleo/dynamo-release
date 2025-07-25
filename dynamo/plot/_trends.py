

import contextlib
import inspect
import functools
import itertools
import os
import types
import warnings
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Callable, Literal, Optional, TypeVar, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import enum
from ..tools._enum import ModeEnum

class GamLinkFunction(ModeEnum):
    IDENTITY = enum.auto()
    LOGIT = enum.auto()
    INVERSE = enum.auto()
    LOG = enum.auto()
    INV_SQUARED = enum.auto()


class GamDistribution(ModeEnum):
    NORMAL = enum.auto()
    BINOMIAL = enum.auto()
    POISSON = enum.auto()
    GAMMA = enum.auto()
    GAUSSIAN = enum.auto()
    INV_GAUSS = enum.auto()


import collections



def _filter_kwargs(_fn: Callable, **kwargs: Any) -> dict:
    """Filter keyword arguments.

    Parameters
    ----------
    _fn
        Function for which to filter keyword arguments.
    kwargs
        Keyword arguments to filter

    Returns
    -------
    dict
        Filtered keyword arguments for the given function.
    """
    sig = inspect.signature(_fn).parameters
    return {k: v for k, v in kwargs.items() if k in sig}



def trends(
        adata,
        pseudotime_key,
        color_key,
        gene,
        clusters=None,
        link='log',
        distribution='gamma',
        spline_order=3,
        n_knots=6,
        lam=3,
        max_lam=1000,
        ax=None,
        figsize=(4,4),
        fontsize=14,
        show=False,
        legend=True,
        legend_kwargs={},
):
    """
    Plot the trends of a gene along a pseudotime trajectory.
    
    Parameters
    ----------
    adata: AnnData object
        The AnnData object containing the data.
    pseudotime_key: str
        The key in adata.obs that contains the pseudotime values.
    color_key: str
        The key in adata.obs that contains the cluster labels.
    gene: str
        The gene to plot.
    clusters: list, optional
        The clusters to plot. If None, all clusters will be plotted.
    link: str, optional
        The link function to use. Default is 'log'.
    distribution: str, optional
        The distribution to use. Default is 'gamma'.
    spline_order: int, optional
        The order of the spline. Default is 3.
    n_knots: int, optional
        The number of knots to use. Default is 6.
    lam: float, optional
        Regularization parameter for the GAM model. Default is 3.
        If optimization fails, this will be automatically increased.
    max_lam: float, optional
        Maximum regularization parameter to try. Default is 1000.
    ax: matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.
    figsize: tuple, optional
        The size of the figure. Default is (4, 4).
    fontsize: int, optional
        The font size for the plot. Default is 14.
    show: bool, optional
        Whether to show the plot. Default is False.
    legend_kwargs: dict, optional
        Additional keyword arguments for the legend. Default is {}.

    Returns
    -------
    ax: matplotlib.axes.Axes
        The axes with the plot.
    """

    from pygam import GAM as pGAM
    from pygam import (
        ExpectileGAM,
        GammaGAM,
        InvGaussGAM,
        LinearGAM,
        LogisticGAM,
        PoissonGAM,
        s,
    )
    from pygam.utils import OptimizationError
    
    _gams = collections.defaultdict(
        lambda: pGAM,
        {
            (GamDistribution.NORMAL, GamLinkFunction.IDENTITY): LinearGAM,
            (GamDistribution.BINOMIAL, GamLinkFunction.LOGIT): LogisticGAM,
            (GamDistribution.POISSON, GamLinkFunction.LOG): PoissonGAM,
            (GamDistribution.GAMMA, GamLinkFunction.LOG): GammaGAM,
            (GamDistribution.INV_GAUSS, GamLinkFunction.LOG): InvGaussGAM,
        },
    )


    distribution = GamDistribution(distribution)
    link = GamLinkFunction(link)

    gam = _gams[distribution, link]
    max_iter=2000


    filtered_kwargs = _filter_kwargs(gam.__init__)
    filtered_kwargs["link"] = link
    filtered_kwargs["distribution"] = distribution

    idx=adata.obs.sort_values(pseudotime_key).index
    x_test=adata.obs.loc[idx,pseudotime_key].values
    w_test=np.ones(len(x_test))

    if ax is None:
        fig,ax=plt.subplots(1,1,figsize = figsize)
    else:
        ax=ax

    if type(adata.uns[f'{color_key}_colors']) == dict:
        color_dict=adata.uns[f'{color_key}_colors']
    elif type(adata.uns[f'{color_key}_colors']) == list:
        color_dict = {}
        for i, ct in enumerate(adata.obs[color_key].cat.categories):
            color_dict[ct] = adata.uns[f'{color_key}_colors'][i]
    else:
        ov_color=['#7CBB5F','#368650','#A499CC','#5E4D9A','#78C2ED','#866017', '#9F987F','#E0DFED',
        '#EF7B77', '#279AD7','#F0EEF0', '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', '#FCBC10',
        '#EAEFC5', '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48', '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']
        color_dict = {}
        for i, ct in enumerate(adata.obs[color_key].cat.categories):
            color_dict[ct] = ov_color[i]

    if clusters is None:
        clusters=list(adata.obs[color_key].cat.categories)

    legend_handles = []
    for ct in clusters:
        idx=adata.obs.loc[adata.obs[color_key] == ct].sort_values(pseudotime_key).index
        x_train=adata.obs.loc[idx,pseudotime_key].values
        w_train=np.ones(len(x_train))
        y_train=adata[idx,gene].to_df().values.reshape(-1)
        
        # Try fitting with increasing regularization until successful
        current_lam = lam
        model_fitted = False
        
        while current_lam <= max_lam and not model_fitted:
            try:
                term = s(
                    0,
                    spline_order=spline_order,
                    n_splines=n_knots,
                    **_filter_kwargs(s, **{**{"lam": current_lam, "penalties": ["derivative", "l2"]},}),
                )

                model = gam(
                    term,
                    max_iter=max_iter,
                    verbose=False,
                    **_filter_kwargs(gam.__init__, **filtered_kwargs),
                )
                
                model.fit(
                    x_train,
                    y_train,
                    weights=w_train
                )
                model_fitted = True
                
            except OptimizationError:
                current_lam *= 3  # Increase regularization
                if current_lam > max_lam:
                    warnings.warn(f"Could not fit GAM model for cluster {ct} with gene {gene}. "
                                f"Optimization failed even with maximum regularization {max_lam}. "
                                f"Skipping this cluster.")
                    break
                
        if not model_fitted:
            continue  # Skip this cluster if we couldn't fit the model
            
        _y_train = model.predict(x_train)
        _y_train = np.squeeze(_y_train)
        _conf_int = model.confidence_intervals(x_train,0.95)

        ax.plot(x_train, _y_train, label=ct, color=color_dict[ct])
        ax.fill_between(
            x_train.squeeze(),
            _conf_int[:, 0],
            _conf_int[:, 1],
            alpha=0.5,
            color=color_dict[ct],
            linestyle="--",
        )
        # 创建圆点的 legend handle（Line2D 类型）
        handle = mlines.Line2D([], [], color=color_dict[ct], marker='o', linestyle='None', label=ct)
        legend_handles.append(handle)
        ax.set_xticklabels(ax.get_xticklabels(),fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels(),fontsize=fontsize)

        
    ax.grid(False)
    ax.set_title(f'{gene}',fontsize=12)
    ax.set_xlabel(pseudotime_key,fontsize=fontsize)
    ax.set_ylabel('Expression',fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    
    

    if legend_kwargs == {}:
        legend_kwargs={
            'bbox_to_anchor':(1.05,1),
            'loc':'upper left',
        }
    if legend:
        plt.legend(handles=legend_handles, 
               fontsize=fontsize,**legend_kwargs)

    

    if show:
        plt.show()
        return None
    else:
        return ax