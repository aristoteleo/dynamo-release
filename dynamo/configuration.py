import warnings
from typing import List, Optional, Tuple, Union

import colorcet
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata._core.anndata import AnnData
from cycler import cycler
from matplotlib import cm, colors, rcParams

from .dynamo_logger import main_debug, main_info


class DynamoAdataKeyManager:
    VAR_GENE_MEAN_KEY = "pp_gene_mean"
    VAR_GENE_VAR_KEY = "pp_gene_variance"
    VAR_GENE_HIGHLY_VARIABLE_KEY = "gene_highly_variable"
    VAR_GENE_HIGHLY_VARIABLE_SCORES = "gene_highly_variable_scores"
    VAR_USE_FOR_PCA = "use_for_pca"

    # a set of preprocessing keys to label dataset properties
    UNS_PP_KEY = "pp"
    UNS_PP_HAS_SPLICING = "has_splicing"
    UNS_PP_TKEY = "time"
    UNS_PP_HAS_LABELING = "has_labeling"
    UNS_PP_HAS_PROTEIN = "has_protein"
    UNS_PP_SPLICING_LABELING = "splicing_labeling"
    UNS_PP_PEARSON_RESIDUAL_NORMALIZATION = "pearson_residuals_normalization_params"

    # obsp adjacency matrix string constants
    OBSP_ADJ_MAT_DIST = "distances"
    OBSP_ADJ_MAT_CONNECTIVITY = "connectivities"

    # special key names frequently used in dynamo
    X_LAYER = "X"
    PROTEIN_LAYER = "protein"
    X_PCA = "X_pca"

    def gen_new_layer_key(layer_name, key, sep="_") -> str:
        """utility function for returning a new key name for a specific layer. By convention layer_name should not have the separator as the last character."""
        if layer_name == "":
            return key
        if layer_name[-1] == sep:
            return layer_name + key
        return sep.join([layer_name, key])

    def gen_layer_pp_key(*keys):
        """Generate dynamo style keys for adata.uns[pp][key0_key1_key2...]"""
        return "_".join(keys)

    def gen_layer_X_key(key):
        """Generate dynamo style keys for adata.layer[X_*], used later in dynamics"""
        return DynamoAdataKeyManager.gen_new_layer_key("X", key)

    def is_layer_X_key(key):
        return key[:2] == "X_"

    def gen_layer_pearson_residual_key(layer: str):
        """Generate dynamo style keys for adata.uns[pp][key0_key1_key2...]"""
        return DynamoAdataKeyManager.gen_layer_pp_key(
            layer, DynamoAdataKeyManager.UNS_PP_PEARSON_RESIDUAL_NORMALIZATION
        )

    def select_layer_data(adata: AnnData, layer: str, copy=False) -> pd.DataFrame:
        """This utility provides a unified interface for selecting layer data.

        The default layer is X layer in adata with shape n_obs x n_var. For protein data it selects adata.obsm["protein"]
        as specified by dynamo convention (the number of proteins are generally less than detected genes `n_var`).
        For other layer data, select data based on layer key with shape n_obs x n_var.
        """
        if layer is None:
            layer = DynamoAdataKeyManager.X_LAYER
        res_data = None
        if layer == DynamoAdataKeyManager.X_LAYER:
            res_data = adata.X
        elif layer == DynamoAdataKeyManager.PROTEIN_LAYER:
            res_data = adata.obsm["protein"] if "protein" in adata.obsm_keys() else None
        else:
            res_data = adata.layers[layer]
        if copy:
            return res_data.copy()
        return res_data

    def set_layer_data(adata: AnnData, layer: str, vals: np.array, var_indices: np.array = None):
        if var_indices is None:
            var_indices = slice(None)
        if layer == DynamoAdataKeyManager.X_LAYER:
            adata.X[:, var_indices] = vals
        elif layer in adata.layers:
            adata.layers[layer][:, var_indices] = vals
        else:
            # layer does not exist in adata
            # ignore var_indices and set values as a new layer
            adata.layers[layer] = vals

    def check_if_layer_exist(adata: AnnData, layer: str) -> bool:
        if layer == DynamoAdataKeyManager.X_LAYER:
            # assume always exist
            return True
        if layer == DynamoAdataKeyManager.PROTEIN_LAYER:
            return DynamoAdataKeyManager.PROTEIN_LAYER in adata.obsm

        return layer in adata.layers

    def get_available_layer_keys(adata, layers="all", remove_pp_layers=True, include_protein=True):
        """Get the list of available layers' keys. If `layers` is set to all, return a list of all available layers; if `layers` is set to a list, then the intersetion of available layers and `layers` will be returned."""
        layer_keys = list(adata.layers.keys())
        if layers is None:  # layers=adata.uns["pp"]["experiment_layers"], in calc_sz_factor
            layers = "X"
        if remove_pp_layers:
            layer_keys = [i for i in layer_keys if not i.startswith("X_")]

        if "protein" in adata.obsm.keys() and include_protein:
            layer_keys.extend(["X", "protein"])
        else:
            layer_keys.extend(["X"])
        res_layers = layer_keys if layers == "all" else list(set(layer_keys).intersection(list(layers)))
        res_layers = list(set(res_layers).difference(["matrix", "ambiguous", "spanning"]))
        return res_layers

    def allowed_layer_raw_names():
        only_splicing = ["spliced", "unspliced"]
        only_labeling = ["new", "total"]
        splicing_and_labeling = ["uu", "ul", "su", "sl"]
        return only_splicing, only_labeling, splicing_and_labeling

    def get_raw_data_layers(adata: AnnData) -> str:
        only_splicing, only_labeling, splicing_and_labeling = DKM.allowed_layer_raw_names()
        # select layers in adata to be normalized
        res = only_splicing + only_labeling + splicing_and_labeling
        res = set(res).intersection(adata.layers.keys()).union("X")
        res = list(res)
        return res

    def allowed_X_layer_names():
        only_splicing = ["X_spliced", "X_unspliced"]
        only_labeling = ["X_new", "X_total"]
        splicing_and_labeling = ["X_uu", "X_ul", "X_su", "X_sl"]

        return only_splicing, only_labeling, splicing_and_labeling

    def init_uns_pp_namespace(adata: AnnData):
        adata.uns[DynamoAdataKeyManager.UNS_PP_KEY] = {}

    def get_excluded_layers(X_total_layers: bool = False, splicing_total_layers: bool = False) -> List:
        """Get a list of excluded layers based on the provided arguments.

        When splicing_total_layers is False, the function normalize spliced and unspliced RNA separately using each
        layer's size factors. When X_total_layers is False, the function normalize X (normally it corresponds to the
        spliced RNA or total RNA for a conventional scRNA-seq or labeling scRNA-seq) using its own size factor.

        Args:
            X_total_layers: whether to also normalize adata.X by size factor from total RNA.
            splicing_total_layers: whether to also normalize spliced / unspliced layers by size factor from total RNA.

        Returns:
            The list of layers to be excluded.
        """
        excluded_layers = []
        if not X_total_layers:
            excluded_layers.extend(["X"])
        if not splicing_total_layers:
            excluded_layers.extend(["spliced", "unspliced"])
        return excluded_layers

    def aggregate_layers_into_total(
        _adata: AnnData,
        layers: Union[str, List[str]] = "all",
        total_layers: Optional[List[str]] = None,
        extend_layers: bool = True,
    ) -> Tuple[Optional[List[str]], Union[str, List[str]]]:
        """Create a total layer in adata by aggregating multiple layers.

        The size factor normalization function is able to calculate size factors from customized layers. Given list
        of total_layers, this helper function will calculate a temporary `_total_` layer.

        Args:
            _adata: the Anndata object.
            layers: the layer(s) to be normailized in the normailzation function.
            total_layers: the layer(s) to sum up to get the total mRNA. For example, ["spliced", "unspliced"],
                ["uu", "ul", "su", "sl"] or ["new", "old"], etc.
            extend_layers: whether to extend the `_total_` layer to the list of layers.

        Returns:
            The tuple contains total layers and layers. Anndata object will be updated with `_total_` layer.
        """
        if not isinstance(total_layers, list):
            total_layers = [total_layers]
        if len(set(total_layers).difference(_adata.layers.keys())) == 0:
            total = None
            for t_key in total_layers:
                total = _adata.layers[t_key] if total is None else total + _adata.layers[t_key]
            _adata.layers["_total_"] = total
            if extend_layers:
                layers.extend(["_total_"])
        return total_layers, layers


# TODO discuss alias naming convention
DKM = DynamoAdataKeyManager


class DynamoVisConfig:
    def set_default_mode(background="white"):
        set_figure_params("dynamo", background=background)


class DynamoAdataConfig:
    """dynamo anndata object config class holding static variables to change behaviors of functions globally."""

    # set the adata store mode.
    # saving memory or storing more results
    # modes: full, succinct
    data_store_mode = None

    # save config for recipe_* functions
    recipe_keep_filtered_genes = None
    recipe_keep_raw_layers = None
    recipe_keep_filtered_cells = None

    # save config for recipe_monocle
    recipe_monocle_keep_filtered_genes = None
    recipe_monocle_keep_filtered_cells = None
    recipe_monocle_keep_raw_layers = None

    dynamics_del_2nd_moments = None
    recipe_del_2nd_moments = None

    # add str variables to store key name string here
    (
        RECIPE_KEEP_FILTERED_CELLS_KEY,
        RECIPE_KEEP_FILTERED_GENES_KEY,
        RECIPE_KEEP_RAW_LAYERS_KEY,
        RECIPE_MONOCLE_KEEP_FILTERED_CELLS_KEY,
        RECIPE_MONOCLE_KEEP_FILTERED_GENES_KEY,
        RECIPE_MONOCLE_KEEP_RAW_LAYERS_KEY,
        DYNAMICS_DEL_2ND_MOMENTS_KEY,
        RECIPE_DEL_2ND_MOMENTS_KEY,
    ) = [
        "keep_filtered_cells_key",
        "keep_filtered_genes_key",
        "keep_raw_layers_key",
        "recipe_monocle_keep_filtered_cells_key",
        "recipe_monocle_keep_filtered_genes_key",
        "recipe_monocle_keep_raw_layers_key",
        "dynamics_del_2nd_moments_key",
        "recipe_del_2nd_moments",
    ]

    # config_key_to_values contains _key to values for config values
    config_key_to_values = None

    def use_default_var_if_none(val, key, replace_val=None):
        """if `val` is equal to `replace_val`, then a config value will be returned according to `key` stored in dynamo configuration. Otherwise return the original `val` value.

        Parameters
        ----------
            val :
                The input value to check against.
            key :
                `key` stored in the dynamo configuration. E.g DynamoAdataConfig.RECIPE_MONOCLE_KEEP_RAW_LAYERS_KEY
            replace_val :
                the target value to replace, by default None

        Returns
        -------
            `val` or config value set in DynamoAdataConfig according to the method description above.

        """
        if not key in DynamoAdataConfig.config_key_to_values:
            assert KeyError("Config %s not exist in DynamoAdataConfig." % (key))
        if val == replace_val:
            config_val = DynamoAdataConfig.config_key_to_values[key]
            main_info("%s is None. Using default value from DynamoAdataConfig: %s=%s" % (key, key, config_val))
            return config_val
        return val

    def update_data_store_mode(mode):
        DynamoAdataConfig.data_store_mode = mode

        # default succinct for recipe*, except for recipe_monocle
        DynamoAdataConfig.recipe_keep_filtered_genes = False
        DynamoAdataConfig.recipe_keep_raw_layers = False
        DynamoAdataConfig.recipe_keep_filtered_cells = False
        DynamoAdataConfig.recipe_del_2nd_moments = True

        if DynamoAdataConfig.data_store_mode == "succinct":
            DynamoAdataConfig.recipe_monocle_keep_filtered_genes = False
            DynamoAdataConfig.recipe_monocle_keep_filtered_cells = False
            DynamoAdataConfig.recipe_monocle_keep_raw_layers = False
            DynamoAdataConfig.dynamics_del_2nd_moments = True
        elif DynamoAdataConfig.data_store_mode == "full":
            DynamoAdataConfig.recipe_monocle_keep_filtered_genes = True
            DynamoAdataConfig.recipe_monocle_keep_filtered_cells = True
            DynamoAdataConfig.recipe_monocle_keep_raw_layers = True
            DynamoAdataConfig.dynamics_del_2nd_moments = False
        else:
            raise NotImplementedError

        DynamoAdataConfig.config_key_to_values = {
            DynamoAdataConfig.RECIPE_KEEP_FILTERED_CELLS_KEY: DynamoAdataConfig.recipe_keep_filtered_cells,
            DynamoAdataConfig.RECIPE_KEEP_FILTERED_GENES_KEY: DynamoAdataConfig.recipe_keep_filtered_genes,
            DynamoAdataConfig.RECIPE_KEEP_RAW_LAYERS_KEY: DynamoAdataConfig.recipe_keep_raw_layers,
            DynamoAdataConfig.RECIPE_MONOCLE_KEEP_FILTERED_CELLS_KEY: DynamoAdataConfig.recipe_monocle_keep_filtered_cells,
            DynamoAdataConfig.RECIPE_MONOCLE_KEEP_FILTERED_GENES_KEY: DynamoAdataConfig.recipe_monocle_keep_filtered_genes,
            DynamoAdataConfig.RECIPE_MONOCLE_KEEP_RAW_LAYERS_KEY: DynamoAdataConfig.recipe_monocle_keep_raw_layers,
            DynamoAdataConfig.DYNAMICS_DEL_2ND_MOMENTS_KEY: DynamoAdataConfig.dynamics_del_2nd_moments,
            DynamoAdataConfig.RECIPE_DEL_2ND_MOMENTS_KEY: DynamoAdataConfig.recipe_del_2nd_moments,
        }


def update_data_store_mode(mode):
    DynamoAdataConfig.update_data_store_mode(mode)


# create cmap
zebrafish_colors = [
    "#4876ff",
    "#85C7F2",
    "#cd00cd",
    "#911eb4",
    "#000080",
    "#808080",
    "#008080",
    "#ffc125",
    "#262626",
    "#3cb44b",
    "#ff4241",
    "#b77df9",
]

zebrafish_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("zebrafish", zebrafish_colors)

fire_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("fire", colorcet.fire)
darkblue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("darkblue", colorcet.kbc)
darkgreen_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("darkgreen", colorcet.kgy)
darkred_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "darkred", colors=colorcet.linear_kry_5_95_c72[:192], N=256
)
darkpurple_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("darkpurple", colorcet.linear_bmw_5_95_c89)
# add gkr theme for velocity
div_blue_black_red_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div_blue_black_red", colorcet.diverging_gkr_60_10_c40
)
# add RdBu_r theme for velocity
div_blue_red_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div_blue_red", colorcet.diverging_bwr_55_98_c37
)
# add glasbey_bw for cell annotation in white background
glasbey_white_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("glasbey_white", colorcet.glasbey_bw_minc_20)
# add glasbey_bw_minc_20_maxl_70 theme for cell annotation in dark background
glasbey_dark_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "glasbey_dark", colorcet.glasbey_bw_minc_20_maxl_70
)

# register cmap
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if "zebrafish" not in matplotlib.colormaps():
        plt.register_cmap("zebrafish", zebrafish_cmap)
    if "fire" not in matplotlib.colormaps():
        plt.register_cmap("fire", fire_cmap)
    if "darkblue" not in matplotlib.colormaps():
        plt.register_cmap("darkblue", darkblue_cmap)
    if "darkgreen" not in matplotlib.colormaps():
        plt.register_cmap("darkgreen", darkgreen_cmap)
    if "darkred" not in matplotlib.colormaps():
        plt.register_cmap("darkred", darkred_cmap)
    if "darkpurple" not in matplotlib.colormaps():
        plt.register_cmap("darkpurple", darkpurple_cmap)
    if "div_blue_black_red" not in matplotlib.colormaps():
        plt.register_cmap("div_blue_black_red", div_blue_black_red_cmap)
    if "div_blue_red" not in matplotlib.colormaps():
        plt.register_cmap("div_blue_red", div_blue_red_cmap)
    if "glasbey_white" not in matplotlib.colormaps():
        plt.register_cmap("glasbey_white", glasbey_white_cmap)
    if "glasbey_dark" not in matplotlib.colormaps():
        plt.register_cmap("glasbey_dark", glasbey_dark_cmap)


_themes = {
    "fire": {
        "cmap": "fire",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "fire",
    },
    "viridis": {
        "cmap": "viridis",
        "color_key_cmap": "Spectral",
        "background": "white",
        "edge_cmap": "gray",
    },
    "inferno": {
        "cmap": "inferno",
        "color_key_cmap": "Spectral",
        "background": "black",
        "edge_cmap": "gray",
    },
    "blue": {
        "cmap": "Blues",
        "color_key_cmap": "tab20",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "red": {
        "cmap": "Reds",
        "color_key_cmap": "tab20b",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "green": {
        "cmap": "Greens",
        "color_key_cmap": "tab20c",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "darkblue": {
        "cmap": "darkblue",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "darkred",
    },
    "darkred": {
        "cmap": "darkred",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "darkblue",
    },
    "darkgreen": {
        "cmap": "darkgreen",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "darkpurple",
    },
    "div_blue_black_red": {
        "cmap": "div_blue_black_red",
        "color_key_cmap": "div_blue_black_red",
        "background": "black",
        "edge_cmap": "gray_r",
    },
    "div_blue_red": {
        "cmap": "div_blue_red",
        "color_key_cmap": "div_blue_red",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "glasbey_dark": {
        "cmap": "glasbey_dark",
        "color_key_cmap": "glasbey_dark",
        "background": "black",
        "edge_cmap": "gray",
    },
    "glasbey_white_zebrafish": {
        "cmap": "zebrafish",
        "color_key_cmap": "zebrafish",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "glasbey_white": {
        "cmap": "glasbey_white",
        "color_key_cmap": "glasbey_white",
        "background": "white",
        "edge_cmap": "gray_r",
    },
}

# https://github.com/vega/vega/wiki/Scales#scale-range-literals
cyc_10 = list(map(colors.to_hex, cm.tab10.colors))
cyc_20 = list(map(colors.to_hex, cm.tab20c.colors))
zebrafish_256 = list(map(colors.to_hex, zebrafish_colors))


# ideally let us convert the following ggplot theme for Nature publisher group into matplotlib.rcParams
# nm_theme <- function() {
#   theme(strip.background = element_rect(colour = 'white', fill = 'white')) +
#     theme(panel.border = element_blank(), axis.line = element_line()) +
#     theme(panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank()) +
#     theme(panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank()) +
#     theme(panel.background = element_rect(fill='white')) +
#     #theme(text = element_text(size=6)) +
#     theme(axis.text.y=element_text(size=6)) +
#     theme(axis.text.x=element_text(size=6)) +
#     theme(axis.title.y=element_text(size=6)) +
#     theme(axis.title.x=element_text(size=6)) +
#     theme(panel.border = element_blank(), axis.line = element_line(size = .1), axis.ticks = element_line(size = .1)) +
#     theme(legend.position = "none") +
#     theme(strip.text.x = element_text(colour="black", size=6)) +
#     theme(strip.text.y = element_text(colour="black", size=6)) +
#     theme(legend.title = element_text(colour="black", size = 6)) +
#     theme(legend.text = element_text(colour="black", size = 6)) +
#     theme(plot.margin=unit(c(0,0,0,0), "lines"))
# }


def dyn_theme(background="white"):
    # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mpl-data/stylelib/dark_background.mplstyle

    if background == "black":
        rcParams.update(
            {
                "lines.color": "w",
                "patch.edgecolor": "w",
                "text.color": "w",
                "axes.facecolor": background,
                "axes.edgecolor": "white",
                "axes.labelcolor": "w",
                "xtick.color": "w",
                "ytick.color": "w",
                "figure.facecolor": background,
                "figure.edgecolor": background,
                "savefig.facecolor": background,
                "savefig.edgecolor": background,
                "grid.color": "w",
                "axes.grid": False,
            }
        )
    else:
        rcParams.update(
            {
                "lines.color": "k",
                "patch.edgecolor": "k",
                "text.color": "k",
                "axes.facecolor": background,
                "axes.edgecolor": "black",
                "axes.labelcolor": "k",
                "xtick.color": "k",
                "ytick.color": "k",
                "figure.facecolor": background,
                "figure.edgecolor": background,
                "savefig.facecolor": background,
                "savefig.edgecolor": background,
                "grid.color": "k",
                "axes.grid": False,
            }
        )


def config_dynamo_rcParams(
    background="white",
    prop_cycle=zebrafish_256,
    fontsize=8,
    color_map=None,
    frameon=None,
):
    """Configure matplotlib.rcParams to dynamo defaults (based on ggplot style and scanpy).

    Parameters
    ----------
        background: `str` (default: `white`)
            The background color of the plot. By default we use the white ground
            which is suitable for producing figures for publication. Setting it to `black` background will
            be great for presentation.
        prop_cycle: `list` (default: zebrafish_256)
            A list with hex color codes
        fontsize: float (default: 6)
            Size of font
        color_map: `plt.cm` or None (default: None)
            Color map
        frameon: `bool` or None (default: None)
            Whether to have frame for the figure.
    Returns
    -------
        Nothing but configure the rcParams globally.
    """

    # from http://www.huyng.com/posts/sane-color-scheme-for-matplotlib/

    rcParams["patch.linewidth"] = 0.5
    rcParams["patch.facecolor"] = "348ABD"  # blue
    rcParams["patch.edgecolor"] = "EEEEEE"
    rcParams["patch.antialiased"] = True

    rcParams["font.size"] = 10.0

    rcParams["axes.facecolor"] = "E5E5E5"
    rcParams["axes.edgecolor"] = "white"
    rcParams["axes.linewidth"] = 1
    rcParams["axes.grid"] = True
    # rcParams['axes.titlesize'] =  "x-large"
    # rcParams['axes.labelsize'] = "large"
    rcParams["axes.labelcolor"] = "555555"
    rcParams["axes.axisbelow"] = True  # grid/ticks are below elements (e.g., lines, text)

    # rcParams['axes.prop_cycle'] = cycler('color', ['E24A33', '348ABD', '988ED5', '777777', 'FBC15E', '8EBA42', 'FFB5B8'])
    # # E24A33 : red
    # # 348ABD : blue
    # # 988ED5 : purple
    # # 777777 : gray
    # # FBC15E : yellow
    # # 8EBA42 : green
    # # FFB5B8 : pink

    # rcParams['xtick.color'] = "555555"
    rcParams["xtick.direction"] = "out"

    # rcParams['ytick.color'] = "555555"
    rcParams["ytick.direction"] = "out"

    rcParams["grid.color"] = "white"
    rcParams["grid.linestyle"] = "-"  # solid line

    rcParams["figure.facecolor"] = "white"
    rcParams["figure.edgecolor"] = "white"  # 0.5

    # the following code is modified from scanpy
    # https://github.com/theislab/scanpy/blob/178a0981405ba8ccfd5031eb15bc07b3a45d2730/scanpy/plotting/_rcmod.py

    # dpi options (mpl default: 100, 100)
    rcParams["figure.dpi"] = 100
    rcParams["savefig.dpi"] = 300

    # figure (default: 0.125, 0.96, 0.15, 0.91)
    rcParams["figure.figsize"] = (6, 4)
    rcParams["figure.subplot.left"] = 0.18
    rcParams["figure.subplot.right"] = 0.96
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.91

    # lines (defaults:  1.5, 6, 1)
    rcParams["lines.linewidth"] = 1.5  # the line width of the frame
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1

    # font
    rcParams["font.sans-serif"] = [
        "Arial",
        "sans-serif",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
    ]
    fontsize = fontsize
    labelsize = 0.90 * fontsize

    # fonsizes (default: 10, medium, large, medium)
    rcParams["font.size"] = fontsize
    rcParams["legend.fontsize"] = labelsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.labelsize"] = labelsize

    # legend (default: 1, 1, 2, 0.8)
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1
    rcParams["legend.handlelength"] = 0.5
    rcParams["legend.handletextpad"] = 0.4

    # color cycle
    rcParams["axes.prop_cycle"] = cycler(color=prop_cycle)  # use tab20c by default

    # lines
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.facecolor"] = "white"

    # ticks (default: k, k, medium, medium)
    rcParams["xtick.color"] = "k"
    rcParams["ytick.color"] = "k"
    rcParams["xtick.labelsize"] = labelsize
    rcParams["ytick.labelsize"] = labelsize

    # axes grid (default: False, #b0b0b0)
    rcParams["axes.grid"] = False
    rcParams["grid.color"] = ".8"

    # color map
    rcParams["image.cmap"] = "RdBu_r" if color_map is None else color_map

    dyn_theme(background)

    # frame (default: True)
    frameon = False if frameon is None else frameon
    global _frameon
    _frameon = frameon


def set_figure_params(
    dynamo=True,
    background="white",
    fontsize=8,
    figsize=(6, 4),
    dpi=None,
    dpi_save=None,
    frameon=None,
    vector_friendly=True,
    color_map=None,
    format="pdf",
    transparent=False,
    ipython_format="png2x",
):
    """Set resolution/size, styling and format of figures.
       This function is adapted from: https://github.com/theislab/scanpy/blob/f539870d7484675876281eb1c475595bf4a69bdb/scanpy/_settings.py
    Arguments
    ---------
        dynamo: `bool` (default: `True`)
            Init default values for :obj:`matplotlib.rcParams` suited for dynamo.
        background: `str` (default: `white`)
            The background color of the plot. By default we use the white ground
            which is suitable for producing figures for publication. Setting it to `black` background will
            be great for presentation.
        fontsize: `[float, float]` or None (default: `6`)
        figsize: `(float, float)` (default: `(6.5, 5)`)
            Width and height for default figure size.
        dpi: `int` or None (default: `None`)
            Resolution of rendered figures - this influences the size of figures in notebooks.
        dpi_save: `int` or None (default: `None`)
            Resolution of saved figures. This should typically be higher to achieve
            publication quality.
        frameon: `bool` or None (default: `None`)
            Add frames and axes labels to scatter plots.
        vector_friendly: `bool` (default: `True`)
            Plot scatter plots using `png` backend even when exporting as `pdf` or `svg`.
        color_map: `str` (default: `None`)
            Convenience method for setting the default color map.
        format: {'png', 'pdf', 'svg', etc.} (default: 'pdf')
            This sets the default format for saving figures: `file_format_figs`.
        transparent: `bool` (default: `False`)
            Save figures with transparent back ground. Sets `rcParams['savefig.transparent']`.
        ipython_format : list of `str` (default: 'png2x')
            Only concerns the notebook/IPython environment; see
            `IPython.core.display.set_matplotlib_formats` for more details.
    """

    try:
        import IPython

        if isinstance(ipython_format, str):
            ipython_format = [ipython_format]
        IPython.display.set_matplotlib_formats(*ipython_format)
    except Exception:
        pass

    from matplotlib import rcParams

    global _vector_friendly, file_format_figs

    _vector_friendly = vector_friendly
    file_format_figs = format

    if dynamo:
        config_dynamo_rcParams(background=background, fontsize=fontsize, color_map=color_map)
    if figsize is not None:
        rcParams["figure.figsize"] = figsize

    if dpi is not None:
        rcParams["figure.dpi"] = dpi
    if dpi_save is not None:
        rcParams["savefig.dpi"] = dpi_save
    if transparent is not None:
        rcParams["savefig.transparent"] = transparent
    if frameon is not None:
        global _frameon
        _frameon = frameon


def reset_rcParams():
    """Reset `matplotlib.rcParams` to defaults."""
    from matplotlib import rcParamsDefault

    rcParams.update(rcParamsDefault)


def set_pub_style(scaler=1):
    """formatting helper function that can be used to save publishable figures"""
    set_figure_params("dynamo", background="white")
    matplotlib.use("cairo")
    matplotlib.rcParams.update({"font.size": 4 * scaler})
    params = {
        "font.size": 4 * scaler,
        "legend.fontsize": 4 * scaler,
        "legend.handlelength": 0.5 * scaler,
        "axes.labelsize": 6 * scaler,
        "axes.titlesize": 6 * scaler,
        "xtick.labelsize": 6 * scaler,
        "ytick.labelsize": 6 * scaler,
        "axes.titlepad": 1 * scaler,
        "axes.labelpad": 1 * scaler,
    }
    matplotlib.rcParams.update(params)


def set_pub_style_mpltex():
    """formatting helper function based on mpltex package that can be used to save publishable figures"""
    set_figure_params("dynamo", background="white")
    matplotlib.use("cairo")
    # the following code is adapted from https://github.com/liuyxpp/mpltex
    # latex_preamble = r"\usepackage{siunitx}\sisetup{detect-all}\usepackage{helvet}\usepackage[eulergreek,EULERGREEK]{sansmath}\sansmath"
    params = {
        "font.family": "sans-serif",
        "font.serif": ["Times", "Computer Modern Roman"],
        "font.sans-serif": [
            "Arial",
            "sans-serif",
            "Helvetica",
            "Computer Modern Sans serif",
        ],
        "font.size": 4,
        # "text.usetex": True,
        # "text.latex.preamble": latex_preamble,  # To force LaTeX use Helvetica
        # "axes.prop_cycle": default_color_cycler,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "axes.linewidth": 1,
        "figure.subplot.left": 0.125,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.1,
        "figure.subplot.top": 0.95,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        # "savefig.bbox": "tight",
        # this will crop white spaces around images that will make
        # width/height no longer the same as the specified one.
        "legend.fontsize": 4,
        "legend.frameon": False,
        "legend.numpoints": 1,
        "legend.handlelength": 0.5,
        "legend.scatterpoints": 1,
        "legend.labelspacing": 0.5,
        "legend.markerscale": 0.9,
        "legend.handletextpad": 0.5,  # pad between handle and text
        "legend.borderaxespad": 0.5,  # pad between legend and axes
        "legend.borderpad": 0.5,  # pad between legend and legend content
        "legend.columnspacing": 1,  # pad between each legend column
        # "text.fontsize" : 4,
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "lines.linewidth": 1,
        "lines.markersize": 4,
        # "lines.markeredgewidth": 0,
        # 0 will make line-type markers, such as "+", "x", invisible
        # Revert some properties to mpl v1 which is more suitable for publishing
        "axes.autolimit_mode": "round_numbers",
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "xtick.direction": "in",
        "xtick.top": True,
        "ytick.direction": "in",
        "ytick.right": True,
        "axes.titlepad": 1,
        "axes.labelpad": 1,
    }
    matplotlib.rcParams.update(params)


# initialize DynamoSaveConfig and DynamoVisConfig mode defaults
DynamoAdataConfig.update_data_store_mode("full")
main_debug("setting visualization default mode in dynamo. Your customized matplotlib settings might be overwritten.")
DynamoVisConfig.set_default_mode()
