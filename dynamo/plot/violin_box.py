import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import issparse

def violin_box(adata, keys, groupby, ax=None, 
               figsize=(4,4), show=True, 
               max_strip_points=200):
    import colorcet
    
    # 获取 y 数据
    y = None
    if not adata.raw is None and keys in adata.raw.var_names:
        y = adata.raw[:, keys].X
    elif keys in adata.obs.columns:
        y = adata.obs[keys].values
    elif keys in adata.var_names:
        y = adata[:, keys].X
    else:
        raise ValueError(f'{keys} not found in adata.raw.var_names, adata.var_names, or adata.obs.columns')
    
    if issparse(y):
        y = y.toarray().reshape(-1)
    else:
        y = y.reshape(-1)
    
    # 获取 x 数据
    x = adata.obs[groupby].values.reshape(-1)
    
    # 创建绘图数据
    plot_data = pd.DataFrame({groupby: x, keys: y})
    
    # 创建图形和轴
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 获取或设置颜色
    if f'{groupby}_colors' not in adata.uns or adata.uns[f'{groupby}_colors'] is None:
        colors = ['#%02x%02x%02x' % tuple([int(k * 255) for k in i]) for i in colorcet.glasbey_bw_minc_20_maxl_70]
        adata.uns[f'{groupby}_colors'] = colors[:len(adata.obs[groupby].unique())]
    
    # 绘制小提琴图
    sns.violinplot(x=groupby, y=keys, data=plot_data, hue=groupby, dodge=False,
                   palette=adata.uns[f'{groupby}_colors'], scale="width", inner=None, ax=ax)
    
    # 调整小提琴图
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
    
    # 绘制箱线图
    sns.boxplot(x=groupby, y=keys, data=plot_data, saturation=1, showfliers=False,
                width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
    
    # 限制 stripplot 的数据点数量
    if len(plot_data) > max_strip_points:
        plot_data = plot_data.sample(max_strip_points)
    
    # 绘制 stripplot
    old_len_collections = len(ax.collections)
    sns.stripplot(x=groupby, y=keys, data=plot_data, hue=groupby,
                  palette=adata.uns[f'{groupby}_colors'], dodge=False, ax=ax)
    
    # 调整 stripplot 点的位置
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.xticks(rotation=90)
    
    if show:
        plt.show()
    
    return ax