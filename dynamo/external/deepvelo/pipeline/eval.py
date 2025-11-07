# this is the evaluation pipeline
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from anndata import AnnData

from ..utils import cross_boundary_correctness, velocity_confidence
from ..plot import compare_plot
from ..tool import stats_test

PathLike = Union[str, Path]


def evaluate(
    result_adatas: Dict[str, AnnData],
    metrics: List[str] = [
        "direction_score",
        "overall_consistency",
        "celltype_consistency",
    ],
    cluster_edges: Optional[List[Tuple[str]]] = None,
    cluster_key: str = "clusters",
    vkey: str = "velocity",
    basis: str = "umap",
    save_dir: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """
    Evaluate metrics and compare on provided adatas that contain the results.

    Args:
        result_adatas (dict): dictionary of adatas that contain the results.
        metrics (list): list of metrics to evaluate. Available metrics: `direction_score`,
            `overall_consistency`, `celltype_consistency`. Default: all metrics.
        cluster_edges (list): list of tuples of cluster names that are considered as
            boundary clusters. Required if `direction_score` is in `metrics`.
        cluster_key (str): key of cluster annotation in `adata.obs`, default: `clusters`.
        vkey (str): key of velocity in `adata.layers`, default: `velocity`.
        save_dir (str or Path): directory to save the plots. If None, do not save plots.

    Returns:
        dict: dictionary of evaluation results.
    """

    if "direction_score" in metrics:
        assert cluster_edges is not None

    use_methods = list(result_adatas.keys())
    adata_list = [result_adatas[method] for method in use_methods]
    eval_results = {}

    if "direction_score" in metrics:
        eval_results["direction_score"] = {}
        for method in use_methods:
            cbcs, avg_cbc = cross_boundary_correctness(
                result_adatas[method],
                cluster_key,
                vkey,
                cluster_edges,
                x_emb_key=basis,  # or Ms
            )
            print(
                f"Average cross-boundary correctness of {method}: {avg_cbc:.2f}\n", cbcs
            )
            all_cbcs_ = result_adatas[method].uns["raw_direction_scores"]
            eval_results["direction_score"][method] = {
                "mean": all_cbcs_.mean(),
                "std": all_cbcs_.std(),
            }

        ax_hist, ax_stat = compare_plot(
            *adata_list,
            labels=list(result_adatas.keys()),
            data=[adata.uns["raw_direction_scores"] for adata in adata_list],
            ylabel="Direction scores",
        )
        if save_dir is not None:
            ax_hist.get_figure().savefig(save_dir / "direction_score_hist.png", dpi=300)
            ax_stat.get_figure().savefig(save_dir / "direction_score_comp.png", dpi=300)
        _, pval = stats_test(*(ad.uns["raw_direction_scores"] for ad in adata_list))
        eval_results["direction_score"]["pval"] = pval

        # # recompute on basis
        # eval_results[f"{basis}_direction_score"] = {}
        # for method in use_methods:
        #     cbcs, avg_cbc = cross_boundary_correctness(
        #         result_adatas[method],
        #         cluster_key,
        #         vkey,
        #         cluster_edges,
        #         x_emb_key=basis,
        #         output_key_prefix=f"{basis}_",
        #     )
        #     print(
        #         f"Average cross-boundary correctness of {method} on {basis}: {avg_cbc:.2f}\n",
        #         cbcs,
        #     )
        #     eval_results[f"{basis}_direction_score"][method] = avg_cbc

        # ax_hist, ax_stat = compare_plot(
        #     *adata_list,
        #     labels=list(result_adatas.keys()),
        #     data=[adata.uns[f"{basis}_raw_direction_scores"] for adata in adata_list],
        #     ylabel=f"{basis.upper()} direction scores",
        # )
        # if save_dir is not None:
        #     ax_hist.get_figure().savefig(
        #         save_dir / f"{basis}_direction_score_hist.png", dpi=300
        #     )
        #     ax_stat.get_figure().savefig(
        #         save_dir / f"{basis}_direction_score_comp.png", dpi=300
        #     )
        # _, pval = stats_test(
        #     *(ad.uns[f"{basis}_raw_direction_scores"] for ad in adata_list)
        # )
        # eval_results[f"{basis}_direction_score"]["pval"] = pval

    if "overall_consistency" in metrics:
        # Compare consistency score
        eval_results["overall_consistency"] = {}
        for method in use_methods:
            velocity_confidence(result_adatas[method], vkey=vkey, method="cosine")
            mean_cosine = result_adatas[method].obs[f"{vkey}_confidence_cosine"].mean()
            std_cosine = result_adatas[method].obs[f"{vkey}_confidence_cosine"].std()
            eval_results["overall_consistency"][method] = {
                "mean": mean_cosine,
                "std": std_cosine,
            }
        ax_hist, ax_stat = compare_plot(*adata_list, labels=list(result_adatas.keys()))
        if save_dir is not None:
            ax_hist.get_figure().savefig(
                save_dir / "overall_consistency_hist.png", dpi=300
            )
            ax_stat.get_figure().savefig(
                save_dir / "overall_consistency_comp.png", dpi=300
            )
        _, pval = stats_test(
            *(ad.obs[f"{vkey}_confidence_cosine"] for ad in adata_list)
        )
        eval_results["overall_consistency"]["pval"] = pval

    if "celltype_consistency" in metrics:
        eval_results["celltype_consistency"] = {}
        # cosine similarity, compute within Celltype
        for method in use_methods:
            velocity_confidence(
                result_adatas[method], vkey=vkey, method="cosine", scope_key=cluster_key
            )
            res_cosine = result_adatas[method].obs[f"{vkey}_confidence_cosine"]
            if res_cosine.isna().sum() > 0:
                warnings.warn(
                    f"NaN values found in adata.obs[{vkey}_confidence_cosine]. "
                    "NaN values will be removed for calculating the average."
                )
                res_cosine = res_cosine.dropna()
            eval_results["celltype_consistency"][method] = {
                "mean": res_cosine.mean(),
                "std": res_cosine.std(),
            }
        ax_hist, ax_stat = compare_plot(
            *adata_list,
            labels=list(result_adatas.keys()),
            ylabel="Celltype-wise consistency",
        )
        if save_dir is not None:
            ax_hist.get_figure().savefig(
                save_dir / "celltype_consistency_hist.png", dpi=300
            )
            ax_stat.get_figure().savefig(
                save_dir / "celltype_consistency_comp.png", dpi=300
            )
        _, pval = stats_test(
            *(ad.obs[f"{vkey}_confidence_cosine"].dropna() for ad in adata_list)
        )
        eval_results["celltype_consistency"]["pval"] = pval

    if save_dir is not None:
        with open(save_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=4)

    return eval_results
