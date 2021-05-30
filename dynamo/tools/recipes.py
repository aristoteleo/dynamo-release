import numpy as np
from .moments import moments
from .connectivity import neighbors, normalize_knn_graph
from .dynamics import dynamics
from .dimension_reduction import reduceDimension
from .cell_velocities import cell_velocities
from .utils import set_transition_genes
from ..configuration import keep_filtered_genes

# add recipe_csc_data()


def recipe_kin_data(
    adata,
    tkey=None,
    reset_X=True,
    X_total_layers=False,
    splicing_total_layers=False,
    n_top_genes=1000,
    keep_filtered_cells=False,
    keep_filtered_genes=keep_filtered_genes,
    keep_raw_layers=False,
    del_2nd_moments=True,
    ekey="M_t",
    vkey="velocity_T",
    basis="umap",
    rm_kwargs={},
):
    """An analysis recipe that properly pre-processes different layers for an kinetics experiment with both labeling and
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
        del_2nd_moments: `bool` (default: `False`)
            Whether to remove second moments or covariances. Default it is `False` so this avoids recalculating 2nd
            moments or covariance but it may take a lot memory when your dataset is big. Set this to `True` when your
            data is huge (like > 25, 000 cells or so) to reducing the memory footprint. Argument used for `dynamics`
            function.
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
    from ..preprocessing.utils import pca, detect_datatype

    has_splicing, has_labeling, splicing_labeling, _ = detect_datatype(adata)

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

        pca(adata, CM[:, valid_ind], pca_key="X_spliced_pca")
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
    keep_filtered_cells=False,
    keep_filtered_genes=keep_filtered_genes,
    keep_raw_layers=False,
    del_2nd_moments=True,
    fraction_for_deg=False,
    ekey="M_s",
    vkey="velocity_S",
    basis="umap",
    rm_kwargs={},
):
    """An analysis recipe that properly pre-processes different layers for a degradation experiment with both
    labeling and splicing data. Functions need to be updated.

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
        del_2nd_moments: `bool` (default: `False`)
            Whether to remove second moments or covariances. Default it is `False` so this avoids recalculating 2nd
            moments or covariance but it may take a lot memory when your dataset is big. Set this to `True` when your
            data is huge (like > 25, 000 cells or so) to reducing the memory footprint. Argument used for `dynamics`
            function.
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
    from ..preprocessing.utils import pca, detect_datatype

    has_splicing, has_labeling, splicing_labeling, _ = detect_datatype(adata)

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
        pca(adata, CM[:, valid_ind], pca_key="X_total_pca")
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
    keep_filtered_cells=False,
    keep_filtered_genes=keep_filtered_genes,
    keep_raw_layers=False,
    del_2nd_moments=True,
    ekey="M_t",
    vkey="velocity_T",
    basis="umap",
    rm_kwargs={},
):
    """An analysis recipe that properly pre-processes different layers for an kinetics experiment with both labeling and
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
        del_2nd_moments: `bool` (default: `False`)
            Whether to remove second moments or covariances. Default it is `False` so this avoids recalculating 2nd
            moments or covariance but it may take a lot memory when your dataset is big. Set this to `True` when your
            data is huge (like > 25, 000 cells or so) to reducing the memory footprint. Argument used for `dynamics`
            function.
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
    from ..preprocessing.utils import pca, detect_datatype

    has_splicing, has_labeling, splicing_labeling, _ = detect_datatype(adata)

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

        pca(adata, CM[:, valid_ind], pca_key="X_spliced_pca")
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


# add one-shot recipe for getting absolute spliced/total RNA velocity
