import numpy as np

from ..configuration import DynamoAdataConfig
from ..preprocessing.utils import pca_monocle
from .cell_velocities import cell_velocities
from .connectivity import neighbors, normalize_knn_graph
from .dimension_reduction import reduceDimension
from .dynamics import dynamics
from .moments import moments
from .utils import set_transition_genes

# add recipe_csc_data()


def recipe_kin_data(
    adata,
    tkey=None,
    reset_X=True,
    X_total_layers=False,
    splicing_total_layers=False,
    n_top_genes=1000,
    keep_filtered_cells=None,
    keep_filtered_genes=None,
    keep_raw_layers=None,
    del_2nd_moments=None,
    ekey="M_t",
    vkey="velocity_T",
    basis="umap",
    rm_kwargs={},
):
    """An analysis recipe that properly pre-processes different layers for an kinetics experiment with both labeling and
    splicing or only labeling data.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that stores data for the the kinetics experiment, must include `uu, ul, su, sl` four
            different layers.
        tkey: `str` or None (default: None)
            The column key for the labeling time  of cells in .obs. Used for labeling based scRNA-seq data (will also
            support for conventional scRNA-seq data). Note that `tkey` will be saved to adata.uns['pp']['tkey'] and used
            in `dyn.tl.dynamics` in which when `group` is None, `tkey` will also be used for calculating  1st/2st moment
            or covariance. We recommend to use hour as the unit of `time`.
        reset_X: bool (default: `False`)
            Whether do you want to let dynamo reset `adata.X` data based on layers stored in your experiment. One
            critical functionality of dynamo is about visualizing RNA velocity vector flows which requires proper data
            into which the high dimensional RNA velocity vectors will be projected.
            (1) For `kinetics` experiment, we recommend the use of `total` layer as `adata.X`;
            (2) For `degradation/conventional` experiment scRNA-seq, we recommend using `splicing` layer as `adata.X`.
            Set `reset_X` to `True` to set those default values if you are not sure.
        splicing_total_layers: bool (default `False`)
            Whether to also normalize spliced / unspliced layers by size factor from total RNA. Paramter to
            `recipe_monocle` function.
        X_total_layers: bool (default `False`)
            Whether to also normalize adata.X by size factor from total RNA. Paramter to `recipe_monocle` function.
        n_top_genes: `int` (default: `1000`)
            How many top genes based on scoring method (specified by sort_by) will be selected as feature genes.
            Arguments required by the `recipe_monocle` function.
        keep_filtered_cells: `bool` (default: `False`)
            Whether to keep genes that don't pass the filtering in the returned adata object. Used in `recipe_monocle`.
        keep_filtered_genes: `bool` (default: `False`)
            Whether to keep genes that don't pass the filtering in the returned adata object. Used in `recipe_monocle`.
        keep_raw_layers: `bool` (default: `False`)
            Whether to keep layers with raw measurements in the returned adata object. Used in `recipe_monocle`.
       del_2nd_moments: `bool` (default: `None`)
            Whether to remove second moments or covariances. Default it is `None` rgument used for `dynamics` function.
         tkey: `str` (default: `time`)
            The column key for the time label of cells in .obs. Used for  the "kinetic" model.
            mode  with labeled data. When `group` is None, `tkey` will also be used for calculating  1st/2st moment or
            covariance. `{tkey}` column must exist in your adata object and indicates the labeling time period.
            Parameters required for `dynamics` function.
        ekey: str or None (optional, default None)
            The dictionary key that corresponds to the gene expression in the layer attribute. By default, ekey and vkey
            will be automatically detected from the adata object. Parameters required by `cell_velocities`.
        vkey: str or None (optional, default None)
            The dictionary key that corresponds to the estimated velocity values in the layers attribute. Parameters
            required by `cell_velocities`
        basis: int (optional, default `umap`)
            The dictionary key that corresponds to the reduced dimension in `.obsm` attribute. Can be `X_spliced_umap`
            or `X_total_umap`, etc. Parameters required by `cell_velocities`
        rm_kwargs: `dict` or None (default: `None`)
            Other Parameters passed into the pp.recipe_monocle function.

    Returns
    -------
        An updated adata object that went through a proper and typical time-resolved RNA velocity analysis.
    """
    from ..preprocessing import recipe_monocle
    from ..preprocessing.utils import detect_experiment_datatype, pca_monocle

    keep_filtered_cells = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_cells, DynamoAdataConfig.RECIPE_KEEP_FILTERED_CELLS_KEY
    )
    keep_filtered_genes = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_genes, DynamoAdataConfig.RECIPE_KEEP_FILTERED_GENES_KEY
    )
    keep_raw_layers = DynamoAdataConfig.use_default_var_if_none(
        keep_raw_layers, DynamoAdataConfig.RECIPE_KEEP_RAW_LAYERS_KEY
    )
    del_2nd_moments = DynamoAdataConfig.use_default_var_if_none(
        del_2nd_moments, DynamoAdataConfig.RECIPE_DEL_2ND_MOMENTS_KEY
    )

    has_splicing, has_labeling, splicing_labeling, _ = detect_experiment_datatype(adata)

    if has_splicing and has_labeling and splicing_labeling:
        layers = ["X_new", "X_total", "X_uu", "X_ul", "X_su", "X_sl"]
    elif has_labeling:
        layers = ["X_new", "X_total"]

    if not has_labeling:
        raise Exception(
            "This recipe is only applicable to kinetics experiment datasets that have "
            "labeling data (at least either with `'uu', 'ul', 'su', 'sl'` or `'new', 'total'` "
            "layers."
        )

    if has_splicing and has_labeling:
        # new, total (and uu, ul, su, sl if existed) layers will be normalized with size factor calculated with total
        # layers spliced / unspliced layers will be normalized independently.
        recipe_monocle(
            adata,
            tkey=tkey,
            experiment_type="kin",
            reset_X=reset_X,
            X_total_layers=X_total_layers,
            splicing_total_layers=splicing_total_layers,
            n_top_genes=n_top_genes,
            total_layers=True,
            keep_filtered_cells=keep_filtered_cells,
            keep_filtered_genes=keep_filtered_genes,
            keep_raw_layers=keep_raw_layers,
            **rm_kwargs,
        )
        tkey = adata.uns["pp"]["tkey"]
        # first calculate moments for labeling data relevant layers using total based connectivity graph
        moments(adata, group=tkey, layers=layers)

        # then we want to calculate moments for spliced and unspliced layers based on connectivity graph from spliced
        # data.
        # first get X_spliced based pca embedding
        CM = np.log1p(adata[:, adata.var.use_for_pca].layers["X_spliced"].A)
        cm_genesums = CM.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()

        pca_monocle(adata, CM[:, valid_ind], pca_key="X_spliced_pca")
        # then get neighbors graph based on X_spliced_pca
        neighbors(adata, X_data=adata.obsm["X_spliced_pca"], layer="X_spliced")
        # then normalize neighbors graph so that each row sums up to be 1
        conn = normalize_knn_graph(adata.obsp["connectivities"] > 0)
        # then calculate moments for spliced related layers using spliced based connectivity graph
        moments(adata, conn=conn, layers=["X_spliced", "X_unspliced"])
        # then perform kinetic estimations with properly preprocessed layers for either the labeling or the splicing
        # data
        dynamics(
            adata,
            model="deterministic",
            est_method="twostep",
            del_2nd_moments=del_2nd_moments,
        )
        # then perform dimension reduction
        reduceDimension(adata, reduction_method=basis)
        # lastly, project RNA velocity to low dimensional embedding.
        cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)
    else:
        recipe_monocle(
            adata,
            tkey=tkey,
            experiment_type="kin",
            reset_X=reset_X,
            X_total_layers=X_total_layers,
            splicing_total_layers=splicing_total_layers,
            n_top_genes=n_top_genes,
            total_layers=True,
            keep_filtered_cells=keep_filtered_cells,
            keep_filtered_genes=keep_filtered_genes,
            keep_raw_layers=keep_raw_layers,
            **rm_kwargs,
        )
        dynamics(
            adata,
            model="deterministic",
            est_method="twostep",
            del_2nd_moments=del_2nd_moments,
        )
        reduceDimension(adata, reduction_method=basis)
        cell_velocities(adata, basis=basis)

    return adata


def recipe_deg_data(
    adata,
    tkey=None,
    reset_X=True,
    X_total_layers=False,
    splicing_total_layers=False,
    n_top_genes=1000,
    keep_filtered_cells=None,
    keep_filtered_genes=None,
    keep_raw_layers=None,
    del_2nd_moments=True,
    fraction_for_deg=False,
    ekey="M_s",
    vkey="velocity_S",
    basis="umap",
    rm_kwargs={},
):
    """An analysis recipe that properly pre-processes different layers for a degradation experiment with both
    labeling and splicing data or only labeling . Functions need to be updated.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that stores data for the the kinetics experiment, must include `uu, ul, su, sl` four
            different layers.
        tkey: `str` or None (default: None)
            The column key for the labeling time  of cells in .obs. Used for labeling based scRNA-seq data (will also
            support for conventional scRNA-seq data). Note that `tkey` will be saved to adata.uns['pp']['tkey'] and used
            in `dyn.tl.dynamics` in which when `group` is None, `tkey` will also be used for calculating  1st/2st moment
            or covariance. We recommend to use hour as the unit of `time`.
        reset_X: bool (default: `False`)
            Whether do you want to let dynamo reset `adata.X` data based on layers stored in your experiment. One
            critical functionality of dynamo is about visualizing RNA velocity vector flows which requires proper data
            into which the high dimensional RNA velocity vectors will be projected.
            (1) For `kinetics` experiment, we recommend the use of `total` layer as `adata.X`;
            (2) For `degradation/conventional` experiment scRNA-seq, we recommend using `splicing` layer as `adata.X`.
            Set `reset_X` to `True` to set those default values if you are not sure.
        splicing_total_layers: bool (default `False`)
            Whether to also normalize spliced / unspliced layers by size factor from total RNA. Paramter to
            `recipe_monocle` function.
        X_total_layers: bool (default `False`)
            Whether to also normalize adata.X by size factor from total RNA. Paramter to `recipe_monocle` function.
        n_top_genes: `int` (default: `1000`)
            How many top genes based on scoring method (specified by sort_by) will be selected as feature genes.
            Arguments required by the `recipe_monocle` function.
        keep_filtered_cells: `bool` (default: `False`)
            Whether to keep genes that don't pass the filtering in the returned adata object. Used in `recipe_monocle`.
        keep_filtered_genes: `bool` (default: `False`)
            Whether to keep genes that don't pass the filtering in the returned adata object. Used in `recipe_monocle`.
        keep_raw_layers: `bool` (default: `False`)
            Whether to keep layers with raw measurements in the returned adata object. Used in `recipe_monocle`.
       del_2nd_moments: `bool` (default: `None`)
            Whether to remove second moments or covariances. Default it is `None` rgument used for `dynamics` function.
         fraction_for_deg: `bool` (default: `False`)
            Whether to use the fraction of labeled RNA instead of the raw labeled RNA to estimate the degradation parameter.
        tkey: `str` (default: `time`)
            The column key for the time label of cells in .obs. Used for  the "kinetic" model.
            mode  with labeled data. When `group` is None, `tkey` will also be used for calculating  1st/2st moment or
            covariance. `{tkey}` column must exist in your adata object and indicates the labeling time period.
            Parameters required for `dynamics` function.
        ekey: str or None (optional, default None)
            The dictionary key that corresponds to the gene expression in the layer attribute. By default, ekey and vkey
            will be automatically detected from the adata object. Parameters required by `cell_velocities`.
        vkey: str or None (optional, default None)
            The dictionary key that corresponds to the estimated velocity values in the layers attribute. Parameters
            required by `cell_velocities`
        basis: int (optional, default `umap`)
            The dictionary key that corresponds to the reduced dimension in `.obsm` attribute. Can be `X_spliced_umap`
            or `X_total_umap`, etc. Parameters required by `cell_velocities`
        rm_kwargs: `dict` or None (default: `None`)
            Other Parameters passed into the pp.recipe_monocle function.

    Returns
    -------
        An updated adata object that went through a proper and typical time-resolved RNA velocity analysis.
    """

    from ..preprocessing import recipe_monocle
    from ..preprocessing.utils import detect_experiment_datatype, pca_monocle

    keep_filtered_cells = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_cells, DynamoAdataConfig.RECIPE_KEEP_FILTERED_CELLS_KEY
    )
    keep_filtered_genes = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_genes, DynamoAdataConfig.RECIPE_KEEP_FILTERED_GENES_KEY
    )
    keep_raw_layers = DynamoAdataConfig.use_default_var_if_none(
        keep_raw_layers, DynamoAdataConfig.RECIPE_KEEP_RAW_LAYERS_KEY
    )

    has_splicing, has_labeling, splicing_labeling, _ = detect_experiment_datatype(adata)

    if has_splicing and has_labeling and splicing_labeling:
        layers = ["X_new", "X_total", "X_uu", "X_ul", "X_su", "X_sl"]
    elif has_labeling:
        layers = ["X_new", "X_total"]

    if not has_labeling:
        raise Exception(
            "This recipe is only applicable to kinetics experiment datasets that have "
            "labeling data (at least either with `'uu', 'ul', 'su', 'sl'` or `'new', 'total'` "
            "layers."
        )

    if has_splicing and has_labeling:
        # new, total (and uu, ul, su, sl if existed) layers will be normalized with size factor calculated with total
        # layers spliced / unspliced layers will be normalized independently.
        recipe_monocle(
            adata,
            tkey=tkey,
            experiment_type="deg",
            reset_X=reset_X,
            X_total_layers=X_total_layers,
            splicing_total_layers=splicing_total_layers,
            n_top_genes=n_top_genes,
            total_layers=True,
            keep_filtered_cells=keep_filtered_cells,
            keep_filtered_genes=keep_filtered_genes,
            keep_raw_layers=keep_raw_layers,
            **rm_kwargs,
        )

        tkey = adata.uns["pp"]["tkey"]
        # first calculate moments for spliced related layers using spliced based connectivity graph
        moments(adata, layers=["X_spliced", "X_unspliced"])

        # then calculate moments for labeling data relevant layers using total based connectivity graph
        # first get X_total based pca embedding
        CM = np.log1p(adata[:, adata.var.use_for_pca].layers["X_total"].A)
        cm_genesums = CM.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()
        pca_monocle(adata, CM[:, valid_ind], pca_key="X_total_pca")
        # then get neighbors graph based on X_spliced_pca
        neighbors(adata, X_data=adata.obsm["X_total_pca"], layer="X_total")
        # then normalize neighbors graph so that each row sums up to be 1
        conn = normalize_knn_graph(adata.obsp["connectivities"] > 0)
        moments(adata, conn=conn, group=tkey, layers=layers)

        # then perform kinetic estimations with properly preprocessed layers for either the labeling or the splicing
        # data
        dynamics(
            adata,
            model="deterministic",
            est_method="twostep",
            del_2nd_moments=del_2nd_moments,
            fraction_for_deg=fraction_for_deg,
        )
        # then perform dimension reduction
        reduceDimension(adata, reduction_method=basis)
        # lastly, project RNA velocity to low dimensional embedding.
        try:
            set_transition_genes(adata)
            cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)
        except BaseException:
            cell_velocities(
                adata,
                min_r2=adata.var.gamma_r2.min(),
                enforce=True,
                vkey=vkey,
                ekey=ekey,
                basis=basis,
            )

    else:
        recipe_monocle(
            adata,
            tkey=tkey,
            experiment_type="deg",
            reset_X=reset_X,
            X_total_layers=X_total_layers,
            splicing_total_layers=splicing_total_layers,
            n_top_genes=n_top_genes,
            total_layers=True,
            keep_filtered_cells=keep_filtered_cells,
            keep_filtered_genes=keep_filtered_genes,
            keep_raw_layers=keep_raw_layers,
            **rm_kwargs,
        )
        dynamics(
            adata,
            model="deterministic",
            del_2nd_moments=del_2nd_moments,
            fraction_for_deg=fraction_for_deg,
        )
        reduceDimension(adata, reduction_method=basis)

    return adata


def recipe_mix_kin_deg_data(
    adata,
    tkey=None,
    reset_X=True,
    X_total_layers=False,
    splicing_total_layers=False,
    n_top_genes=1000,
    keep_filtered_cells=None,
    keep_filtered_genes=None,
    keep_raw_layers=None,
    del_2nd_moments=None,
    ekey="M_t",
    vkey="velocity_T",
    basis="umap",
    rm_kwargs={},
):
    """An analysis recipe that properly pre-processes different layers for an mixture kinetics and degradation
    experiment with both labeling and splicing or only labeling data.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that stores data for the the kinetics experiment, must include `uu, ul, su, sl` four
            different layers.
        tkey: `str` or None (default: None)
            The column key for the labeling time  of cells in .obs. Used for labeling based scRNA-seq data (will also
            support for conventional scRNA-seq data). Note that `tkey` will be saved to adata.uns['pp']['tkey'] and used
            in `dyn.tl.dynamics` in which when `group` is None, `tkey` will also be used for calculating  1st/2st moment
            or covariance. We recommend to use hour as the unit of `time`.
        reset_X: bool (default: `False`)
            Whether do you want to let dynamo reset `adata.X` data based on layers stored in your experiment. One
            critical functionality of dynamo is about visualizing RNA velocity vector flows which requires proper data
            into which the high dimensional RNA velocity vectors will be projected.
            (1) For `kinetics` experiment, we recommend the use of `total` layer as `adata.X`;
            (2) For `degradation/conventional` experiment scRNA-seq, we recommend using `splicing` layer as `adata.X`.
            Set `reset_X` to `True` to set those default values if you are not sure.
        splicing_total_layers: bool (default `False`)
            Whether to also normalize spliced / unspliced layers by size factor from total RNA. Paramter to
            `recipe_monocle` function.
        X_total_layers: bool (default `False`)
            Whether to also normalize adata.X by size factor from total RNA. Paramter to `recipe_monocle` function.
        n_top_genes: `int` (default: `1000`)
            How many top genes based on scoring method (specified by sort_by) will be selected as feature genes.
            Arguments required by the `recipe_monocle` function.
        keep_filtered_cells: `bool` (default: `False`)
            Whether to keep genes that don't pass the filtering in the returned adata object. Used in `recipe_monocle`.
        keep_filtered_genes: `bool` (default: `False`)
            Whether to keep genes that don't pass the filtering in the returned adata object. Used in `recipe_monocle`.
        keep_raw_layers: `bool` (default: `False`)
            Whether to keep layers with raw measurements in the returned adata object. Used in `recipe_monocle`.
       del_2nd_moments: `bool` (default: `None`)
            Whether to remove second moments or covariances. Default it is `None` rgument used for `dynamics` function.
         tkey: `str` (default: `time`)
            The column key for the time label of cells in .obs. Used for  the "kinetic" model.
            mode  with labeled data. When `group` is None, `tkey` will also be used for calculating  1st/2st moment or
            covariance. `{tkey}` column must exist in your adata object and indicates the labeling time period.
            Parameters required for `dynamics` function.
        ekey: str or None (optional, default None)
            The dictionary key that corresponds to the gene expression in the layer attribute. By default, ekey and vkey
            will be automatically detected from the adata object. Parameters required by `cell_velocities`.
        vkey: str or None (optional, default None)
            The dictionary key that corresponds to the estimated velocity values in the layers attribute. Parameters
            required by `cell_velocities`
        basis: int (optional, default `umap`)
            The dictionary key that corresponds to the reduced dimension in `.obsm` attribute. Can be `X_spliced_umap`
            or `X_total_umap`, etc. Parameters required by `cell_velocities`
        rm_kwargs: `dict` or None (default: `None`)
            Other Parameters passed into the pp.recipe_monocle function.

    Returns
    -------
        An updated adata object that went through a proper and typical time-resolved RNA velocity analysis.
    """
    from ..preprocessing import recipe_monocle
    from ..preprocessing.utils import detect_experiment_datatype, pca_monocle

    keep_filtered_cells = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_cells, DynamoAdataConfig.RECIPE_KEEP_FILTERED_CELLS_KEY
    )
    keep_filtered_genes = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_genes, DynamoAdataConfig.RECIPE_KEEP_FILTERED_GENES_KEY
    )
    keep_raw_layers = DynamoAdataConfig.use_default_var_if_none(
        keep_raw_layers, DynamoAdataConfig.RECIPE_KEEP_RAW_LAYERS_KEY
    )
    del_2nd_moments = DynamoAdataConfig.use_default_var_if_none(
        del_2nd_moments, DynamoAdataConfig.RECIPE_DEL_2ND_MOMENTS_KEY
    )

    has_splicing, has_labeling, splicing_labeling, _ = detect_experiment_datatype(adata)

    if has_splicing and has_labeling and splicing_labeling:
        layers = ["X_new", "X_total", "X_uu", "X_ul", "X_su", "X_sl"]
    elif has_labeling:
        layers = ["X_new", "X_total"]

    if not has_labeling:
        raise Exception(
            "This recipe is only applicable to kinetics experiment datasets that have "
            "labeling data (at least either with `'uu', 'ul', 'su', 'sl'` or `'new', 'total'` "
            "layers."
        )

    if has_splicing and has_labeling:
        # new, total (and uu, ul, su, sl if existed) layers will be normalized with size factor calculated with total
        # layers spliced / unspliced layers will be normalized independently.
        recipe_monocle(
            adata,
            tkey=tkey,
            experiment_type="mix_pulse_chase",
            reset_X=reset_X,
            X_total_layers=X_total_layers,
            splicing_total_layers=splicing_total_layers,
            n_top_genes=n_top_genes,
            total_layers=True,
            keep_filtered_cells=keep_filtered_cells,
            keep_filtered_genes=keep_filtered_genes,
            keep_raw_layers=keep_raw_layers,
            **rm_kwargs,
        )
        tkey = adata.uns["pp"]["tkey"]
        # first calculate moments for labeling data relevant layers using total based connectivity graph
        moments(adata, group=tkey, layers=layers)

        # then we want to calculate moments for spliced and unspliced layers based on connectivity graph from spliced
        # data.
        # first get X_spliced based pca embedding
        CM = np.log1p(adata[:, adata.var.use_for_pca].layers["X_spliced"].A)
        cm_genesums = CM.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()

        pca_monocle(adata, CM[:, valid_ind], pca_key="X_spliced_pca")
        # then get neighbors graph based on X_spliced_pca
        neighbors(adata, X_data=adata.obsm["X_spliced_pca"], layer="X_spliced")
        # then normalize neighbors graph so that each row sums up to be 1
        conn = normalize_knn_graph(adata.obsp["connectivities"] > 0)
        # then calculate moments for spliced related layers using spliced based connectivity graph
        moments(adata, conn=conn, layers=["X_spliced", "X_unspliced"])
        # then perform kinetic estimations with properly preprocessed layers for either the labeling or the splicing
        # data
        dynamics(
            adata,
            model="deterministic",
            est_method="twostep",
            del_2nd_moments=del_2nd_moments,
        )
        # then perform dimension reduction
        reduceDimension(adata, reduction_method=basis)
        # lastly, project RNA velocity to low dimensional embedding.
        cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)
    else:
        recipe_monocle(
            adata,
            tkey=tkey,
            experiment_type="mix_pulse_chase",
            reset_X=reset_X,
            X_total_layers=X_total_layers,
            splicing_total_layers=splicing_total_layers,
            n_top_genes=n_top_genes,
            total_layers=True,
            keep_filtered_cells=keep_filtered_cells,
            keep_filtered_genes=keep_filtered_genes,
            keep_raw_layers=keep_raw_layers,
            **rm_kwargs,
        )
        dynamics(
            adata,
            model="deterministic",
            est_method="twostep",
            del_2nd_moments=del_2nd_moments,
        )
        reduceDimension(adata, reduction_method=basis)
        cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)

    return adata


# support using just spliced/unspliced/new/total 4 layers, as well as uu, ul, su, sl layers
def recipe_one_shot_data(
    adata,
    tkey=None,
    reset_X=True,
    X_total_layers=False,
    splicing_total_layers=False,
    n_top_genes=1000,
    keep_filtered_cells=None,
    keep_filtered_genes=None,
    keep_raw_layers=None,
    one_shot_method="sci-fate",
    del_2nd_moments=None,
    ekey="M_t",
    vkey="velocity_T",
    basis="umap",
    rm_kwargs={},
):
    """An analysis recipe that properly pre-processes different layers for an one-shot experiment with both labeling and
    splicing data.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that stores data for the the kinetics experiment, must include `uu, ul, su, sl` four
            different layers.
        tkey: `str` or None (default: None)
            The column key for the labeling time  of cells in .obs. Used for labeling based scRNA-seq data (will also
            support for conventional scRNA-seq data). Note that `tkey` will be saved to adata.uns['pp']['tkey'] and used
            in `dyn.tl.dynamics` in which when `group` is None, `tkey` will also be used for calculating  1st/2st moment
            or covariance. We recommend to use hour as the unit of `time`.
        reset_X: bool (default: `False`)
            Whether do you want to let dynamo reset `adata.X` data based on layers stored in your experiment. One
            critical functionality of dynamo is about visualizing RNA velocity vector flows which requires proper data
            into which the high dimensional RNA velocity vectors will be projected.
            (1) For `kinetics` experiment, we recommend the use of `total` layer as `adata.X`;
            (2) For `degradation/conventional` experiment scRNA-seq, we recommend using `splicing` layer as `adata.X`.
            Set `reset_X` to `True` to set those default values if you are not sure.
        splicing_total_layers: bool (default `False`)
            Whether to also normalize spliced / unspliced layers by size factor from total RNA. Paramter to
            `recipe_monocle` function.
        X_total_layers: bool (default `False`)
            Whether to also normalize adata.X by size factor from total RNA. Paramter to `recipe_monocle` function.
        n_top_genes: `int` (default: `1000`)
            How many top genes based on scoring method (specified by sort_by) will be selected as feature genes.
            Arguments required by the `recipe_monocle` function.
        keep_filtered_cells: `bool` (default: `False`)
            Whether to keep genes that don't pass the filtering in the returned adata object. Used in `recipe_monocle`.
        keep_filtered_genes: `bool` (default: `False`)
            Whether to keep genes that don't pass the filtering in the returned adata object. Used in `recipe_monocle`.
        keep_raw_layers: `bool` (default: `False`)
            Whether to keep layers with raw measurements in the returned adata object. Used in `recipe_monocle`.
        one_shot_method: `str` (default: `sci-fate`)
            The method to use for calculate the absolute labeling and splicing velocity for the one-shot data of use.
        del_2nd_moments: `bool` (default: `None`)
            Whether to remove second moments or covariances. Default it is `None` rgument used for `dynamics` function.
        tkey: `str` (default: `time`)
            The column key for the time label of cells in .obs. Used for  the "kinetic" model.
            mode  with labeled data. When `group` is None, `tkey` will also be used for calculating  1st/2st moment or
            covariance. `{tkey}` column must exist in your adata object and indicates the labeling time period.
            Parameters required for `dynamics` function.
        ekey: str or None (optional, default None)
            The dictionary key that corresponds to the gene expression in the layer attribute. By default, ekey and vkey
            will be automatically detected from the adata object. Parameters required by `cell_velocities`.
        vkey: str or None (optional, default None)
            The dictionary key that corresponds to the estimated velocity values in the layers attribute. Parameters
            required by `cell_velocities`
        basis: int (optional, default `umap`)
            The dictionary key that corresponds to the reduced dimension in `.obsm` attribute. Can be `X_spliced_umap`
            or `X_total_umap`, etc. Parameters required by `cell_velocities`
        rm_kwargs: `dict` or None (default: `None`)
            Other Parameters passed into the pp.recipe_monocle function.

    Returns
    -------
        An updated adata object that went through a proper and typical time-resolved RNA velocity analysis.
    """
    from ..preprocessing import recipe_monocle
    from ..preprocessing.utils import detect_experiment_datatype, pca_monocle

    keep_filtered_cells = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_cells, DynamoAdataConfig.RECIPE_KEEP_FILTERED_CELLS_KEY
    )
    keep_filtered_genes = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_genes, DynamoAdataConfig.RECIPE_KEEP_FILTERED_GENES_KEY
    )
    keep_raw_layers = DynamoAdataConfig.use_default_var_if_none(
        keep_raw_layers, DynamoAdataConfig.RECIPE_KEEP_RAW_LAYERS_KEY
    )
    del_2nd_moments = DynamoAdataConfig.use_default_var_if_none(
        del_2nd_moments, DynamoAdataConfig.RECIPE_DEL_2ND_MOMENTS_KEY
    )

    has_splicing, has_labeling, splicing_labeling, _ = detect_experiment_datatype(adata)

    if has_splicing and has_labeling and splicing_labeling:
        layers = ["X_new", "X_total", "X_uu", "X_ul", "X_su", "X_sl"]
    elif has_labeling:
        layers = ["X_new", "X_total"]

    if not has_labeling:
        raise Exception(
            "This recipe is only applicable to kinetics experiment datasets that have "
            "labeling data (at least either with `'uu', 'ul', 'su', 'sl'` or `'new', 'total'` "
            "layers."
        )

    if has_splicing and has_labeling:
        # new, total (and uu, ul, su, sl if existed) layers will be normalized with size factor calculated with total
        # layers spliced / unspliced layers will be normalized independently.
        recipe_monocle(
            adata,
            tkey=tkey,
            experiment_type="one-shot",
            reset_X=reset_X,
            X_total_layers=X_total_layers,
            splicing_total_layers=splicing_total_layers,
            n_top_genes=n_top_genes,
            total_layers=True,
            keep_filtered_cells=keep_filtered_cells,
            keep_filtered_genes=keep_filtered_genes,
            keep_raw_layers=keep_raw_layers,
            **rm_kwargs,
        )
        tkey = adata.uns["pp"]["tkey"]
        # first calculate moments for labeling data relevant layers using total based connectivity graph
        moments(adata, group=tkey, layers=layers)

        # then we want to calculate moments for spliced and unspliced layers based on connectivity graph from spliced
        # data.
        # first get X_spliced based pca embedding
        CM = np.log1p(adata[:, adata.var.use_for_pca].layers["X_spliced"].A)
        cm_genesums = CM.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()

        pca_monocle(adata, CM[:, valid_ind], pca_key="X_spliced_pca")
        # then get neighbors graph based on X_spliced_pca
        neighbors(adata, X_data=adata.obsm["X_spliced_pca"], layer="X_spliced")
        # then normalize neighbors graph so that each row sums up to be 1
        conn = normalize_knn_graph(adata.obsp["connectivities"] > 0)
        # then calculate moments for spliced related layers using spliced based connectivity graph
        moments(adata, conn=conn, layers=["X_spliced", "X_unspliced"])
        # then perform kinetic estimations with properly preprocessed layers for either the labeling or the splicing
        # data
        dynamics(
            adata,
            model="deterministic",
            one_shot_method=one_shot_method,
            del_2nd_moments=del_2nd_moments,
        )
        # then perform dimension reduction
        reduceDimension(adata, reduction_method=basis)
        # lastly, project RNA velocity to low dimensional embedding.
        cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)
    else:
        recipe_monocle(
            adata,
            tkey=tkey,
            experiment_type="one-shot",
            reset_X=reset_X,
            X_total_layers=X_total_layers,
            splicing_total_layers=splicing_total_layers,
            n_top_genes=n_top_genes,
            total_layers=True,
            keep_filtered_cells=keep_filtered_cells,
            keep_filtered_genes=keep_filtered_genes,
            keep_raw_layers=keep_raw_layers,
            **rm_kwargs,
        )
        dynamics(
            adata,
            model="deterministic",
            one_shot_method=one_shot_method,
            del_2nd_moments=del_2nd_moments,
        )
        reduceDimension(adata, reduction_method=basis)
        cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)

    return adata


def velocity_N(
    adata,
    group=None,
    recalculate_pca=True,
    recalculate_umap=True,
    del_2nd_moments=None,
):
    """use new RNA based pca, umap, for velocity calculation and projection for kinetics or one-shot experiment.

    Note that currently velocity_N function only considers labeling data and removes splicing data if they exist.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that stores data for the the kinetics or one-shot experiment, must include `X_new, X_total`
            layers.
        group: `str` or None (default: None)
            The cell group that will be used to calculate velocity in each separate group. This is useful if your data
            comes from different labeling condition, etc.
        recalculate_pca: `bool` (default: True)
            Whether to recalculate pca with the new RNA data. If setting to be False, you need to make sure the pca is
            already generated via new RNA.
        recalculate_umap: `bool` (default: True)
            Whether to recalculate umap with the new RNA data. If setting to be False, you need to make sure the umap is
            already generated via new RNA.
        del_2nd_moments: `None` or `bool`
            Whether to remove second moments or covariances. Default it is `None` rgument used for `dynamics` function.

    Returns
    -------
        Nothing but the adata object is updated with the low dimensional (umap or pca) velocity projections with the
        new RNA or pca based RNA velocities.
    """

    del_2nd_moments = DynamoAdataConfig.use_default_var_if_none(
        del_2nd_moments, DynamoAdataConfig.RECIPE_DEL_2ND_MOMENTS_KEY
    )

    var_columns = adata.var.columns
    layer_keys = adata.layers.keys()

    # check velocity_N, velocity_T, X_new, X_total
    if not np.all([i in layer_keys for i in ["X_new", "X_total"]]):
        raise Exception(f"The `X_new`, `X_total` has to exist in your data before running velocity_N function.")

    # delete the moments and velocities that generated via total RNA
    for i in ["M_t", "M_tt", "M_n", "M_tn", "M_nn", "velocity_N", "velocity_T"]:
        if i in layer_keys:
            del adata.layers[i]

    # delete the kinetic paraemters that generated via total RNA
    for i in [
        "alpha",
        "beta",
        "gamma",
        "half_life",
        "alpha_b",
        "alpha_r2",
        "gamma_b",
        "gamma_r2",
        "gamma_logLL",
        "delta_b",
        "delta_r2",
        "bs",
        "bf",
        "uu0",
        "ul0",
        "su0",
        "sl0",
        "U0",
        "S0",
        "total0",
        "beta_k",
        "gamma_k",
    ]:
        if i in var_columns:
            del adata.var[i]

    if group is not None:
        group_prefixes = [group + "_" + str(i) + "_" for i in adata.obs[group].unique()]
        for i in group_prefixes:
            for j in [
                "alpha",
                "beta",
                "gamma",
                "half_life",
                "alpha_b",
                "alpha_r2",
                "gamma_b",
                "gamma_r2",
                "gamma_logLL",
                "delta_b",
                "delta_r2",
                "bs",
                "bf",
                "uu0",
                "ul0",
                "su0",
                "sl0",
                "U0",
                "S0",
                "total0",
                "beta_k",
                "gamma_k",
            ]:
                if i + j in var_columns:
                    del adata.var[i + j]

    # now let us first run pca with new RNA
    if recalculate_pca:
        pca_monocle(adata, np.log1p(adata[:, adata.var.use_for_pca].layers["X_new"]), pca_key="X_pca")

    # if there are unspliced / spliced data, delete them for now:
    for i in ["spliced", "unspliced", "X_spliced", "X_unspliced"]:
        if i in layer_keys:
            del adata.layers[i]

    # now redo the RNA velocity analysis with moments generated with pca space of new RNA
    # let us also check whether it is a one-shot or kinetics experiment
    if adata.uns["pp"]["experiment_type"] == "one-shot":
        dynamics(
            adata,
            one_shot_method="sci_fate",
            model="deterministic",
            group=group,
            del_2nd_moments=del_2nd_moments,
        )
    elif adata.uns["pp"]["experiment_type"] == "kin":
        dynamics(
            adata,
            model="deterministic",
            est_method="twostep",
            group=group,
            del_2nd_moments=del_2nd_moments,
        )
    else:
        raise Exception(
            f"velocity_N function only supports either the one-shot or kinetics (kin) metabolic labeling "
            f"experiment."
        )

    # umap based on new RNA
    if recalculate_umap:
        reduceDimension(adata, enforce=True)

    # project new RNA velocity to new RNA pca
    cell_velocities(
        adata,
        basis="pca",
        X=adata.layers["M_n"],
        V=adata.layers["velocity_N"],
        enforce=True,
    )

    # project new RNA velocity to new RNA umap
    cell_velocities(
        adata,
        basis="umap",
        X=adata.layers["M_n"],
        V=adata.layers["velocity_N"],
        enforce=True,
    )
