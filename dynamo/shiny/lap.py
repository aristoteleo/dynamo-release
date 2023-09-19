import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import shiny.experimental as x
from anndata import AnnData
from htmltools import TagList, div
from functools import reduce
from pathlib import Path
from shiny import App, Inputs, Outputs, reactive, Session, render, ui
from sklearn.metrics import roc_curve, auc

from .utils import filter_fig
from ..prediction import GeneTrajectory, least_action, least_action_path
from ..plot import kinetic_heatmap, streamline_plot
from ..plot.utils import get_color_map_from_labels, map2color
from ..tools import neighbors
from ..tools.utils import nearest_neighbors, select_cell
from ..vectorfield import rank_genes


css_path = Path(__file__).parent / "styles.css"


def lap_web_app(input_adata: AnnData, tfs_data: AnnData):
    """The shiny web application of most probable path predictions analyses. The process is equivalent to this tutorial:
    https://dynamo-release.readthedocs.io/en/latest/notebooks/lap_tutorial/lap_tutorial.html

    Args:
        input_adata: the processed anndata object to perform LAP analyses.
        tfs_data: the transcription factors information saved in a txt or csv file. The names of TFs should be saved in
            the `["Symbol"]` column. An example can be found in the `sample_data.py`.
    """
    app_ui = x.ui.page_sidebar(
        x.ui.sidebar(
            ui.include_css(css_path),
            x.ui.accordion(
                x.ui.accordion_panel(
                    div("Run pairwise least action path analyses", class_="bold-title"),
                    x.ui.accordion_panel(
                        div("Streamline Plot", class_="bold-subtitle"),
                        ui.input_text("cells_type_key", "Key of cell type information: ", value="cell_type"),
                        ui.input_text("streamline_basis", "The basis to perform LAP: ", value="umap"),
                        ui.input_action_button(
                            "activate_streamline_plot", "Streamline plot", class_="btn-primary"
                        ),
                        value="Streamline Plot",
                    ),
                    x.ui.accordion_panel(
                        div("Initialization", class_="bold-subtitle"),
                        ui.input_text(
                            "cells_names",
                            "Names of cell types to perform LAP analyses: ",
                            placeholder="e.g. HSC,Meg,Ery,Bas,Mon,Neu",
                        ),
                        ui.input_text(
                            "fps_coordinates",
                            "Identified fixed points coordinates for each cell types: ",
                            placeholder="Enter the coordinates of fixed point."
                        ),
                        ui.input_action_button(
                            "initialize", "Initialize searching", class_="btn-primary"
                        ),
                        value="Initialization",
                    ),
                    x.ui.accordion_panel(
                        div("Run LAP", class_="bold-subtitle"),
                        div("Run pairwise least action path analyses among given cell types", class_="explanation"),
                        ui.input_action_button(
                            "activate_lap", "Run LAP analyses", class_="btn-primary"
                        ),
                        value="Run LAP",
                    ),
                    x.ui.accordion_panel(
                        div("Visualize LAP", class_="bold-subtitle"),
                        ui.input_text(
                            "visualize_keys", "Key of transitions to plot: ", placeholder="e.g. HSC->Meg,HSC->Ery"
                        ),
                        ui.input_action_button(
                            "activate_visualize_lap", "Visualize LAP", class_="btn-primary"
                        ),
                        value="Visualize LAP",
                    ),
                    x.ui.accordion_panel(
                        div("Prepare TFs", class_="bold-subtitle"),
                        div("Load the transcription factors data", class_="explanation"),
                        ui.input_action_button(
                            "activate_prepare_tfs", "Prepare TFs", class_="btn-primary"
                        ),
                        value="Prepare TFs",
                    ),
                    x.ui.accordion_panel(
                        div("TFs barplot", class_="bold-subtitle"),
                        ui.input_text(
                            "cell_type_colormap",
                            "Color of each cell type (random color will be selected if not input): ",
                            placeholder="Enter Color Dict"
                        ),
                        ui.input_action_button(
                            "activate_tfs_barplot", "TFs barplot", class_="btn-primary"
                        ),
                        value="TFs barplot",
                    ),
                    x.ui.accordion_panel(
                        div("Pairwise cell fate heatmap", class_="bold-subtitle"),
                        div(
                            "Heatmap of LAP actions and LAP time matrices of pairwise cell fate conversions",
                            class_="explanation"
                        ),
                        ui.input_action_button(
                            "activate_pairwise_cell_fate_heatmap", "Pairwise cell fate heatmap", class_="btn-primary"
                        ),
                        value="Pairwise cell fate heatmap",
                    ),
                    x.ui.accordion_panel(
                        div("LAP Kinetic Heatmap", class_="bold-subtitle"),
                        div(
                            " Plot the kinetic heatmap of given gene expression kinetics of all transcription factors",
                            class_="explanation"
                        ),
                        ui.input_text("lap_init_cells", "Init cells: ", placeholder="e.g. GGGGGGCGGCCT-JL_10"),
                        ui.input_text("lap_end_cells", "End cells: ", placeholder="e.g. GCAGCGAAGGCA-JL12_0"),
                        ui.input_text("lap_basis", "Basis: ", value="pca"),
                        ui.input_text("lap_adj_key", "Adj key to locate transition matrix: ", value="cosine_transition_matrix"),
                        ui.input_action_button(
                            "activate_lap_kinetic_heatmap", "LAP kinetic heatmap", class_="btn-primary"
                        ),
                        value="LAP Kinetic Heatmap",
                    ),
                    value="Run pairwise least action path analyses",
                ),
                x.ui.accordion_panel(
                    div("Evaluate TF rankings based on LAP analyses", class_="bold-title"),
                    x.ui.accordion_panel(
                        div("Add known TFs", class_="bold-subtitle"),
                        ui.input_text("known_tf_transition", "Target transition: ", placeholder="e.g. HSC->Meg"),
                        ui.input_text("known_tf", "Known TFs: ", placeholder="e.g. GATA1,GATA2,ZFPM1,GFI1B,FLI1,NFE2"),
                        ui.input_text("known_tf_key", "Key to save TFs: ", value="TFs"),
                        ui.input_text("known_tf_rank_key", "Key to save TFs rank: ", value="TFs_rank"),
                        ui.input_action_button(
                            "activate_add_known_tf", "Add", class_="btn-primary"
                        ),
                        value="Add known TFs",
                    ),
                    x.ui.accordion_panel(
                        div("Priority Scores of TFs", class_="bold-subtitle"),
                        ui.input_text("reprog_mat_main_key", "Main Key: ", placeholder="e.g. HSC->Meg"),
                        div(
                            "The 'genes' information will be extracted from "
                            "transition_graph[Transition Key][Genes Key]. ",
                            class_="explanation",
                        ),
                        div(
                            "The 'rank' information will be extracted from "
                            "transition_graph[Transition Key][Rank Key]",
                            class_="explanation",
                        ),
                        ui.input_text("reprog_mat_transition_key", "Transition Key: ", placeholder="e.g. HSC->Meg"),
                        ui.input_text("reprog_mat_genes_key", "Genes Key: ", value="TFs"),
                        ui.input_text("reprog_mat_rank_key", "Rank Key: ", value="TFs_rank"),
                        ui.input_text("reprog_mat_PMID", "PMID: ", placeholder="e.g. 18295580"),
                        ui.input_text("reprog_mat_type", "Type of transition: ", placeholder="e.g. development"),
                        ui.input_action_button(
                            "activate_add_reprog_info", "Add", class_="btn-primary"
                        ),
                        ui.input_text("reprog_query_type", "Query transition type to plot: ", value="transdifferentiation"),
                        ui.input_action_button(
                            "activate_plot_priority_scores", "Plot priority scores for TFs", class_="btn-primary"
                        ),
                        ui.input_text("roc_tf_key", "Key of TFs for ROC plot: ", value="TFs"),
                        ui.input_text("roc_target_transition", "Target transition of ROC plot: ", placeholder="HSC->Bas"),
                        ui.input_action_button(
                            "activate_tf_roc_curve", "ROC curve analyses of TF priorization", class_="btn-primary"
                        ),
                        value="Priority Scores of TFs",
                    ),
                    value="Evaluate TF rankings based on LAP analyses",
                ),
                open=False,
            ),
            width=500,
        ),
        ui.div(
            ui.navset_tab(
                ui.nav(
                    "Run pairwise least action path analyses",
                    div("Streamline plot of given basis and color", class_="bold-subtitle"),
                    x.ui.output_plot("base_streamline_plot"),
                    div("The fixed points representing the typical cell state of these cells", class_="bold-subtitle"),
                    x.ui.output_plot("initialize_searching"),
                    div("LAP result of given transition", class_="bold-subtitle"),
                    x.ui.output_plot("plot_lap"),
                    div("Barplot of the LAP time of given LAPs", class_="bold-subtitle"),
                    x.ui.output_plot("tfs_barplot"),
                    div(
                        "Heatmap of LAP actions and LAP time matrices of pairwise cell fate conversions",
                        class_="bold-subtitle"
                    ),
                    x.ui.output_plot("pairwise_cell_fate_heatmap"),
                    div("Kinetics heatmap of gene expression dynamics along the LAP", class_="bold-subtitle"),
                    x.ui.output_plot("lap_kinetic_heatmap"),
                ),
                ui.nav(
                    "Evaluate TF rankings based on LAP analyses",
                    div(
                        "Visualization of the transition_graph which contains LAP and known TFs information",
                        class_="bold-subtitle",
                    ),
                    x.ui.card(
                        ui.output_text_verbatim("add_known_tf"),
                    ),
                    div(
                        "Visualization of dictionary for converting the rankings of known TFs to a priority score",
                        class_="bold-subtitle",
                    ),
                    x.ui.card(
                        ui.output_text_verbatim("add_reprog_info"),
                    ),
                    div(
                        "Plotting priority scores of known TFs for specific hematopoietic trandifferentiations",
                        class_="bold-subtitle",
                    ),
                    x.ui.output_plot("plot_priority_scores"),
                    div("ROC curve analyses of TF priorization of the LAP predictions", class_="bold-subtitle"),
                    x.ui.output_plot("tf_roc_curve")
                ),
            ),
        ),
    )

    def server(input: Inputs, output: Outputs, session: Session):
        adata = input_adata.copy()
        tfs_names = list(tfs_data["Symbol"])
        cells = reactive.Value[list[np.ndarray]]()
        cells_indices = reactive.Value[list[list[float]]]()
        transition_graph = reactive.Value[dict]()
        t_dataframe = reactive.Value[pd.DataFrame]()
        action_dataframe = reactive.Value[pd.DataFrame]()
        transition_color_dict = reactive.Value[dict]()
        transition_color_dict.set(
            {
                "development": "#2E3192",
                "reprogramming": "#EC2227",
                "transdifferentiation": "#B9519E",
            }
        )
        reprogramming_mat_dict = reactive.Value[dict]()
        reprogramming_mat_dict.set({})
        reprogramming_mat_dataframe_p = reactive.Value[pd.DataFrame]()

        @output
        @render.plot()
        @reactive.event(input.initialize)
        def initialize_searching():
            cells_names = input.cells_names().split(",")
            cell_list = []
            for c in cells_names:
                cell_list.append(select_cell(adata, input.cells_type_key(), c))

            fps_coordinates = [float(num) for num in input.fps_coordinates().split(",")]
            fps_coordinates = [[fps_coordinates[i], fps_coordinates[i + 1]] for i in range(0, len(fps_coordinates), 2)]

            cells_indices_list = []
            for coord in fps_coordinates:
                cells_indices_list.append(nearest_neighbors(coord, adata.obsm["X_" + input.streamline_basis()]))

            cells.set(cell_list)
            cells_indices.set(cells_indices_list)

            plt.scatter(*adata.obsm["X_" + input.streamline_basis()].T)
            for indices in cells_indices_list:
                plt.scatter(*adata[indices[0]].obsm["X_" + input.streamline_basis()].T)

        @output
        @render.plot()
        @reactive.event(input.activate_streamline_plot)
        def base_streamline_plot():
            neighbors(adata, basis=input.streamline_basis(), result_prefix=input.streamline_basis())

            axes_list = streamline_plot(
                adata,
                color=input.cells_type_key().split(","),
                basis=input.streamline_basis(),
                save_show_or_return="return",
            )

            return filter_fig(plt.gcf())

        @reactive.Effect
        @reactive.event(input.activate_lap)
        def _():
            transition_graph_dict = {}
            cell_type = input.cells_names().split(",")
            start_cell_indices = cells_indices()
            end_cell_indices = start_cell_indices
            for i, start in enumerate(start_cell_indices):
                for j, end in enumerate(end_cell_indices):
                    if start is not end:
                        min_lap_t = True if i == 0 else False
                        least_action(
                            adata,
                            [adata.obs_names[start[0]][0]],
                            [adata.obs_names[end[0]][0]],
                            basis="umap",
                            adj_key="X_umap_distances",
                            min_lap_t=min_lap_t,
                            EM_steps=2,
                        )
                        # least_action(adata, basis="umap")
                        lap = least_action(
                            adata,
                            [adata.obs_names[start[0]][0]],
                            [adata.obs_names[end[0]][0]],
                            basis="pca",
                            adj_key="cosine_transition_matrix",
                            min_lap_t=min_lap_t,
                            EM_steps=2,
                        )
                        # dyn.pl.kinetic_heatmap(
                        #     adata,
                        #     basis="pca",
                        #     mode="lap",
                        #     genes=adata.var_names[adata.var.use_for_transition],
                        #     project_back_to_high_dim=True,
                        # )
                        # The `GeneTrajectory` class can be used to output trajectories for any set of genes of interest
                        gtraj = GeneTrajectory(adata)
                        gtraj.from_pca(lap.X, t=lap.t)
                        gtraj.calc_msd()
                        ranking = rank_genes(adata, "traj_msd")

                        print(start, "->", end)
                        genes = ranking[:5]["all"].to_list()
                        arr = gtraj.select_gene(genes)

                        # dyn.pl.multiplot(lambda k: [plt.plot(arr[k, :]), plt.title(genes[k])], np.arange(len(genes)))

                        transition_graph_dict[cell_type[i] + "->" + cell_type[j]] = {
                            "lap": lap,
                            "LAP_umap": adata.uns["LAP_umap"],
                            "LAP_pca": adata.uns["LAP_pca"],
                            "ranking": ranking,
                            "gtraj": gtraj,
                        }
            transition_graph.set(transition_graph_dict)

        @output
        @render.plot()
        @reactive.event(input.activate_visualize_lap)
        def plot_lap():
            paths = input.visualize_keys().split(",")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax = streamline_plot(
                adata,
                basis=input.streamline_basis(),
                save_show_or_return="return",
                ax=ax,
                color=input.cells_type_key().split(","),
                frontier=True,
            )
            ax = ax[0]
            x, y = 0, 1

            # plot paths
            for path in paths:
                lap_dict = transition_graph()[path]["LAP_umap"]
                for prediction, action in zip(lap_dict["prediction"], lap_dict["action"]):
                    ax.scatter(*prediction[:, [x, y]].T, c=map2color(action))
                    ax.plot(*prediction[:, [x, y]].T, c="k")

            return filter_fig(fig)

        @reactive.Effect
        @reactive.event(input.activate_prepare_tfs)
        def _():
            cell_type = input.cells_names().split(",")
            action_df = pd.DataFrame(index=cell_type, columns=cell_type)
            t_df = pd.DataFrame(index=cell_type, columns=cell_type)
            for i, start in enumerate(cells_indices()):
                for j, end in enumerate(cells_indices()):
                    if start is not end:
                        print(cell_type[i] + "->" + cell_type[j], end=",")
                        lap = transition_graph()[cell_type[i] + "->" + cell_type[j]]["lap"]  # lap
                        gtraj = transition_graph()[cell_type[i] + "->" + cell_type[j]]["gtraj"]
                        ranking = transition_graph()[cell_type[i] + "->" + cell_type[j]]["ranking"].copy()
                        ranking["TF"] = [i in tfs_names for i in list(ranking["all"])]
                        genes = ranking.query("TF == True").head(10)["all"].to_list()
                        arr = gtraj.select_gene(genes)
                        action_df.loc[cell_type[i], cell_type[j]] = lap.action_t()[-1]
                        t_df.loc[cell_type[i], cell_type[j]] = lap.t[-1]

            action_dataframe.set(action_df)
            t_dataframe.set(t_df)

        @output
        @render.plot()
        @reactive.event(input.activate_tfs_barplot)
        def tfs_barplot():
            develop_time_df = pd.DataFrame({"integration time": t_dataframe().iloc[0, :].T})
            develop_time_df["lineage"] = input.cells_names().split(",")

            ig, ax = plt.subplots(figsize=(4, 3))

            if input.cell_type_colormap():
                colors = input.cell_type_colormap().split(",")
                dynamo_color_dict = {}
                for i in range(len(develop_time_df["lineage"])):
                    dynamo_color_dict[develop_time_df["lineage"][i]] = colors[i]
            else:
                dynamo_color_dict = get_color_map_from_labels(develop_time_df["lineage"].values)

            sns.barplot(
                y="lineage",
                x="integration time",
                hue="lineage",
                data=develop_time_df.iloc[1:, :],
                dodge=False,
                palette=dynamo_color_dict,
                ax=ax,
            )
            ax.set_ylabel("")
            plt.tight_layout()
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            return filter_fig(ig)

        @output
        @render.plot()
        @reactive.event(input.activate_pairwise_cell_fate_heatmap)
        def pairwise_cell_fate_heatmap():
            action_df = action_dataframe().fillna(0)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))
            ax1 = sns.heatmap(action_df, annot=True, ax=ax1)
            t_df = t_dataframe().fillna(0)
            ax2 = sns.heatmap(t_df, annot=True, ax=ax2)
            return filter_fig(f)

        @output
        @render.plot()
        @reactive.event(input.activate_lap_kinetic_heatmap)
        def lap_kinetic_heatmap():
            init_cells = input.lap_init_cells().split(",")
            end_cells = input.lap_end_cells().split(",")
            lap = least_action(
                adata,
                init_cells=init_cells,
                target_cells=end_cells,
                basis=input.lap_basis(),
                adj_key=input.lap_adj_key(),
            )
            is_human_tfs = [gene in tfs_names for gene in
                            adata.var_names[adata.var.use_for_transition]]
            human_genes = adata.var_names[adata.var.use_for_transition][is_human_tfs]
            sns.set(font_scale=0.8)
            sns_heatmap = kinetic_heatmap(
                adata,
                basis=input.lap_basis(),
                mode="lap",
                figsize=(10, 5),
                genes=human_genes,
                project_back_to_high_dim=True,
                save_show_or_return="return",
                color_map="bwr",
                transpose=True,
                xticklabels=True,
                yticklabels=False
            )

            plt.setp(sns_heatmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
            plt.tight_layout()
            return filter_fig(plt.gcf())

        def format_dict_to_text(dictionary, target_key, indent=0):
            text = ""
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    text += "  " * indent + f"{key}:\n"
                    text += format_dict_to_text(value, target_key=target_key, indent=indent + 1)
                else:
                    text += "  " * indent + f"{key}:"
                    text += "  " + f"{value}\n" if key in target_key else "  " + f"...\n"
            return text

        @output
        @render.text
        @reactive.event(input.activate_add_known_tf)
        def add_known_tf():
            transition = input.known_tf_transition()
            cur_transition_graph = transition_graph()

            ranking = cur_transition_graph[transition]["ranking"]
            ranking["TF"] = [i in tfs_names for i in list(ranking["all"])]
            true_tf_list = list(ranking.query("TF == True")["all"])
            all_tfs = list(ranking.query("TF == True")["all"])
            cur_transition_graph[transition][input.known_tf_key()] = input.known_tf().split(",")

            cur_transition_graph[transition][input.known_tf_rank_key()] = [
                all_tfs.index(key) if key in true_tf_list else -1 for key in cur_transition_graph[transition][input.known_tf_key()]
            ]

            transition_graph.set(cur_transition_graph)

            return format_dict_to_text(cur_transition_graph, [input.known_tf_key()])

        def assign_random_color(new_key: str):
            available_colors = ["#fde725", "#5ec962", "#21918c", "#3b528b", "#440154"]
            dictionary = transition_color_dict()
            available_colors = [c for c in available_colors if c not in dictionary.values()]
            random_color = random.choice(available_colors)
            dictionary[new_key] = random_color
            transition_color_dict.set(dictionary)

        @output
        @render.text
        @reactive.event(input.activate_add_reprog_info)
        def add_reprog_info():
            reprog_dict = reprogramming_mat_dict()
            transition_graph_dict = transition_graph()

            reprog_dict[input.reprog_mat_main_key()] = {
                "genes": transition_graph_dict[input.reprog_mat_transition_key()][input.reprog_mat_genes_key()],
                "rank": transition_graph_dict[input.reprog_mat_transition_key()][input.reprog_mat_rank_key()],
                "PMID": input.reprog_mat_PMID(),
                "type": input.reprog_mat_type(),
            }

            if input.reprog_mat_type() not in transition_color_dict().keys():
                assign_random_color(input.reprog_mat_type())

            reprogramming_mat_dict.set(reprog_dict)

            return format_dict_to_text(reprog_dict, ["genes", "type"])

        @output
        @render.plot()
        @reactive.event(input.activate_plot_priority_scores)
        def plot_priority_scores():
            reprogramming_mat_df = pd.DataFrame(reprogramming_mat_dict())
            all_genes = reduce(lambda a, b: a + b, reprogramming_mat_df.loc["genes", :])
            all_rank = reduce(lambda a, b: a + b, reprogramming_mat_df.loc["rank", :])
            all_keys = np.repeat(
                np.array(list(reprogramming_mat_dict().keys())), [len(i) for i in reprogramming_mat_df.loc["genes", :]]
            )
            all_types = np.repeat(
                np.array([v["type"] for v in reprogramming_mat_dict().values()]),
                [len(i) for i in reprogramming_mat_df.loc["genes", :]],
            )

            reprogramming_mat_df_p = pd.DataFrame({"genes": all_genes, "rank": all_rank, "transition": all_keys, "type": all_types})
            reprogramming_mat_df_p = reprogramming_mat_df_p.query("rank > -1")
            reprogramming_mat_df_p["rank"] /= 133
            reprogramming_mat_df_p["rank"] = 1 - reprogramming_mat_df_p["rank"]
            reprogramming_mat_dataframe_p.set(reprogramming_mat_df_p)

            query = "type == '{}'".format(input.reprog_query_type())
            reprogramming_mat_df_p_subset = reprogramming_mat_df_p.query(query)
            rank = reprogramming_mat_df_p_subset["rank"].values
            transition = reprogramming_mat_df_p_subset["transition"].values
            genes = reprogramming_mat_df_p_subset["genes"].values

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            sns.scatterplot(
                y="transition",
                x="rank",
                data=reprogramming_mat_df_p_subset,
                ec=None,
                hue="type",
                alpha=0.8,
                ax=ax,
                s=50,
                palette=transition_color_dict(),
                clip_on=False,
            )

            for i in range(reprogramming_mat_df_p_subset.shape[0]):
                annote_text = genes[i]  # STK_ID
                ax.annotate(
                    annote_text, xy=(rank[i], transition[i]), xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom"
                )

            plt.axvline(0.8, linestyle="--", lw=0.5)
            ax.set_xlim(0.6, 1.01)
            ax.set_xlabel("")
            ax.set_xlabel("Score")
            ax.set_yticklabels(list(reprogramming_mat_dict().keys())[6:], rotation=0)
            ax.legend().set_visible(False)
            ax.spines.top.set_position(("outward", 10))
            ax.spines.bottom.set_position(("outward", 10))

            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")

            return filter_fig(fig)

        @output
        @render.plot()
        @reactive.event(input.activate_tf_roc_curve)
        def tf_roc_curve():
            all_ranks_dict = {}
            for key, value in transition_graph().items():
                ranking = transition_graph()[key]["ranking"]
                ranking["TF"] = [i in tfs_names for i in list(ranking["all"])]
                ranking = ranking.query("TF == True")
                ranking["known_TF"] = [i in value[input.roc_tf_key()] for i in list(ranking["all"])]
                all_ranks_dict[key.split("->")[0] + "_" + key.split("->")[1] + "_ranking"] = ranking

            all_ranks_df = pd.concat([rank_dict for rank_dict in all_ranks_dict.values()])

            target_ranking = all_ranks_dict[
                input.roc_target_transition().split("->")[0] +
                "_" +
                input.roc_target_transition().split("->")[1] +
                "_ranking"
                ]

            all_ranks_df["priority_score"] = (
                    1 - np.tile(np.arange(target_ranking.shape[0]), len(all_ranks_dict)) / target_ranking.shape[0]
            )

            cls = all_ranks_df["known_TF"].astype(int)
            pred = all_ranks_df["priority_score"]

            fpr, tpr, _ = roc_curve(cls, pred)
            roc_auc = auc(fpr, tpr)

            lw = 0.5
            plt.figure(figsize=(5, 5))
            plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title(cur_guide)
            plt.legend(loc="lower right")
            plt.tight_layout()

            return filter_fig(plt.gcf())

    app = App(app_ui, server, debug=True)
    app.run()