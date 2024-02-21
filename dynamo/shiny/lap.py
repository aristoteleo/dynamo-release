from typing import List, Optional

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from anndata import AnnData
from functools import reduce
from pathlib import Path
from sklearn.metrics import roc_curve, auc

from .utils import filter_fig
from ..prediction import GeneTrajectory, least_action, least_action_path
from ..plot import kinetic_heatmap, streamline_plot
from ..plot.utils import get_color_map_from_labels, map2color
from ..tools import neighbors
from ..tools.utils import nearest_neighbors, select_cell
from ..vectorfield import rank_genes


css_path = Path(__file__).parent / "styles.css"

def lap_web_app(input_adata: AnnData, tfs_data: Optional[AnnData]=None):
    """The shiny web application of most probable path predictions analyses. The process is equivalent to this tutorial:
    https://dynamo-release.readthedocs.io/en/latest/notebooks/lap_tutorial/lap_tutorial.html

    Args:
        input_adata: the processed anndata object to perform LAP analyses.
        tfs_data: the transcription factors information saved in a txt or csv file. The names of TFs should be saved in
            the `["Symbol"]` column. An example can be found in the `sample_data.py`.
    """
    try:
        import shiny.experimental as x
        from htmltools import HTML, TagList, div
        from shiny import App, Inputs, Outputs, reactive, Session, render, ui
        from shiny.plotutils import brushed_points, near_points
    except ImportError:
        raise ImportError("Please install shiny and htmltools before running the web application!")

    app_ui = ui.page_fluid(
        ui.include_css(css_path),
        ui.navset_tab(
            ui.nav(
                "Run pairwise least action path analyses",
                ui.panel_main(
                    div("Most probable path predictions", class_="bold-title"),
                    div(HTML("<br><br>")),
                    div("The least action path (LAP) is a principled method that has previously been used in "
                        "theoretical efforts to predict the most probable path a cell will follow during fate "
                        "transition. Specifically, the optimal path between any two cell states (e.g. the fixed "
                        "point of HSCs and that of megakaryocytes) is searched by variating the continuous path "
                        "connecting the source state to the target while minimizing its action and updating the "
                        "associated transition time. The resultant least action path has the highest transition "
                        "probability and is associated with a particular transition time.", class_="explanation"),
                    div(HTML("<br><br>")),
                    ui.div(
                        x.ui.card(
                            div("Initialization", class_="bold-sectiontitle"),
                            div("Given the group information and basis, we can visualize the projected velocity "
                                "information. The velocity provides us with fundamental insights into cell fate "
                                "transitions.", class_="explanation"),
                            ui.row(
                                ui.column(6, ui.output_ui("selectize_cells_type_key")),
                                ui.column(6, ui.output_ui("selectize_streamline_basis")),
                            ),
                            x.ui.output_plot("base_streamline_plot"),
                            div("In the scatter plot, we can choose the fixed points to initialize the LAP analyses. ",
                                class_="explanation"),
                            x.ui.output_plot("initialize_searching", click=True, dblclick=True, hover=True,
                                             brush=True),

                            ui.row(
                                ui.column(
                                    6,
                                    ui.row(
                                        ui.tags.b("Points near cursor"),
                                        ui.output_table("near_click"),
                                    ),
                                    ui.input_action_button("add_click_pts", "Add", class_="btn-primary"),
                                    ui.row(
                                        ui.tags.b("Points in brush"),
                                        ui.output_table("in_brush"),
                                    ),
                                    ui.input_action_button("add_brush_pts", "Add", class_="btn-primary"),
                                ),
                                ui.column(
                                    6,
                                    ui.tags.b("Identified Cells to initialize the path"),
                                    ui.output_table("fixed_points"),
                                    ui.input_action_button("reset_fixed_points", "Reset", class_="btn-primary"),
                                    ui.input_action_button(
                                        "activate_lap", "Run LAP analyses with identified points",
                                        class_="btn-primary",
                                    ),
                                ),
                            ),
                        ),
                        x.ui.card(
                            div("LAP results", class_="bold-sectiontitle"),
                            div("After calculating LAPs for all possible cell type transition pairs, the results will "
                                "be visualized in this section.", class_="explanation"),
                            div("Barplot of genes' ranking based on the mean squared displacement of the path.",
                                class_="bold-subtitle"),
                            ui.row(
                                ui.column(
                                    6,
                                    ui.input_slider("top_n_genes", "Top N genes to visualize: ", min=0, max=20, value=10),
                                ),
                                ui.column(6, ui.output_ui("selectize_gene_barplot_transition")),
                            ),
                            x.ui.output_plot("genes_barplot"),
                            div("Visualization LAPs for one or more transitions.", class_="bold-subtitle"),
                            ui.row(
                                ui.column(
                                    3,
                                    ui.input_slider(
                                        "n_lap_visualize_transition",
                                        "Number of transitions to visualize: ",
                                        min=1, max=20, value=1),
                                    ui.output_ui("selectize_lap_visualize_transition"),
                                ),
                                ui.column(
                                    9,
                                    x.ui.output_plot("plot_lap"),
                                ),
                            ),
                            div("Barplot of the LAP time starting from given cell type", class_="bold-subtitle"),
                            ui.input_switch("if_global_lap_time_rank", "Display global LAP time", value=False),
                            div("Note: If enabled, the rank of all transitions will be displayed. Else, will rank "
                                "the transitions with given starting cell type.", class_="explanation"),
                            ui.output_ui("selectize_barplot_start_genes"),
                            x.ui.output_plot("tfs_barplot"),
                            div(
                                "Heatmap of LAP actions (left) and LAP time (right) matrices of pairwise cell fate conversions",
                                class_="bold-subtitle"
                            ),
                            x.ui.output_plot("pairwise_cell_fate_heatmap"),
                            div("Kinetics heatmap of gene expression dynamics along the LAP",
                                class_="bold-subtitle"),
                            ui.row(
                                ui.column(
                                    3,
                                    ui.output_ui("selectize_kinetic_heatmap_transition"),
                                    ui.output_ui("selectize_lap_heatmap_basis"),
                                    ui.output_ui("selectize_lap_heatmap_adj_key"),
                                    ui.input_slider("heatmap_n_genes", "number of genes to visualize in the plot: ", min=1, max=200, value=50),
                                ),
                                ui.column(
                                    9,
                                    x.ui.output_plot("lap_kinetic_heatmap"),
                                ),
                            ),
                        ),
                    ),
                ),
            ),

            ui.nav(
                "Evaluate TF rankings based on LAP analyses",
                ui.panel_main(
                    div("Evaluate TF rankings based on LAP analyses.", class_="bold-title"),
                    div(HTML("<br><br>")),
                    div("After we obtained the TFs ranking based on the mean square displacement, we are able to "
                        "evaluate rankings by comparing with known transcription factors that enable the successful "
                        "cell fate conversion.", class_="explanation"),
                    div(HTML("<br><br>")),
                    ui.div(
                        x.ui.card(
                            div("Initialization", class_="bold-sectiontitle"),
                            div(
                                "Visualization of transition information and known TFs",
                                class_="bold-subtitle",
                            ),
                            div("Here we need to manually add known TFs and transition type to all possible transition "
                                "pairs.", class_="explanation"),
                            ui.row(
                                ui.column(
                                    3,
                                    div("Add known TFs", class_="bold-subtitle"),
                                    div("First, choose target transition and input known TFs.", class_="explanation"),
                                    ui.output_ui("selectize_known_tf_transition"),
                                    ui.input_text("known_tf", "Known TFs: ",
                                                  placeholder="e.g. GATA1,GATA2,ZFPM1,GFI1B,FLI1,NFE2"),
                                    div("Next, specify the keys to extract and save the TFs and rank. The TFs and rank "
                                        "will be saved in dictionary[main key][TF key] and "
                                        "dictionary[main key][TF rank key]. We don't need to change the default value "
                                        "unless there are more than one set of known genes to analyze in one "
                                        "transition.", class_="explanation"),
                                    ui.input_text("known_tf_key", "Key to save TFs: ", value="TFs"),
                                    ui.input_text("known_tf_rank_key", "Key to save TFs rank: ", value="TFs_rank"),
                                    ui.output_ui("input_reprog_mat_main_key"),
                                    div(
                                        "Then we can specify the type of transition.",
                                        class_="explanation",
                                    ),
                                    ui.output_ui("selectize_reprog_mat_type"),
                                    ui.input_action_button(
                                        "activate_add_reprog_info", "Add transition info", class_="btn-primary",
                                    ),
                                    div("The known TF dictionary will be visualized on the right. After we add known "
                                        "TFs for all transitions, we can click the following button to start the "
                                        "analyses", class_="explanation"),
                                    ui.input_action_button(
                                        "activate_plot_priority_scores_and_ROC", "Analyze with current TFs",
                                        class_="btn-primary"
                                    ),
                                ),
                                ui.column(
                                    9,
                                    ui.output_text_verbatim("add_reprog_info"),
                                ),
                            ),
                        ),
                        x.ui.card(
                            div("TF evaluation results", class_="bold-sectiontitle"),
                            div(
                                "Plotting priority scores of known TFs for specific transition type",
                                class_="bold-subtitle",
                            ),
                            div("The ranking of known TFs will be converted to a priority score, simply defined as "
                                "1 - ( rank / number of TFs ).", class_="explanation"),
                            ui.output_ui("selectize_reprog_query_type"),
                            x.ui.output_plot("plot_priority_scores"),
                            div("ROC curve analyses of TF priorization of the LAP predictions", class_="bold-subtitle"),
                            div("We can evaluate the TF ranking through ROC of LAP TF prioritization predictions using "
                                "all known genes of all known transitions as the gold standard.", class_="explanation"),
                            ui.input_text("roc_tf_key", "Key of TFs for ROC plot: ", value="TFs"),
                            x.ui.output_plot("tf_roc_curve")
                        ),
                    ),
                ),
            ),
        ),
    )

    def server(input: Inputs, output: Outputs, session: Session):
        adata = input_adata.copy()
        coordinates_df = reactive.Value[pd.DataFrame]()
        initialize_fps_coordinates = reactive.Value[pd.DataFrame]()
        initialize_fps_coordinates.set(None)
        tfs_names = list(tfs_data["Symbol"]) if tfs_data is not None else list(adata.var_names)
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
        @render.ui
        def selectize_cells_type_key():
            return ui.input_selectize(
                        "cells_type_key",
                        "Key representing the group information, most of the time it is related to cell type: ",
                        choices=list(adata.obs.keys()),
                        selected="cell_type",
                    )

        @output
        @render.ui
        def selectize_streamline_basis():
            return ui.input_selectize(
                "streamline_basis",
                "The basis to perform LAP: ",
                choices=[b[2:] if b.startswith("X_") else b for b in list(adata.obsm.keys())],
                selected="umap",
            )

        @output
        @render.table()
        def near_click():
            return near_points(
                coordinates_df().copy(),
                coordinfo=input.initialize_searching_click(),
                xvar="x",
                yvar="y",
            )

        @reactive.Effect
        @reactive.event(input.add_click_pts)
        def _():
            points = near_points(
                coordinates_df().copy(),
                coordinfo=input.initialize_searching_click(),
                xvar="x",
                yvar="y",
            )
            if initialize_fps_coordinates() is None:
                initialize_fps_coordinates.set(points)
            else:
                initialize_fps_coordinates.set(pd.concat([initialize_fps_coordinates(), points]))

        @output
        @render.table()
        def in_brush():
            return brushed_points(
                coordinates_df().copy(),
                input.initialize_searching_brush(),
                xvar="x",
                yvar="y",
            )

        @reactive.Effect
        @reactive.event(input.add_brush_pts)
        def _():
            points = brushed_points(
                coordinates_df().copy(),
                input.initialize_searching_brush(),
                xvar="x",
                yvar="y",
            )
            if initialize_fps_coordinates() is None:
                initialize_fps_coordinates.set(points)
            else:
                initialize_fps_coordinates.set(pd.concat([initialize_fps_coordinates(), points]))

        @output
        @render.plot()
        # @reactive.event(input.add_click_pts, input.add_brush_pts, input.reset_fixed_points)
        def initialize_searching():
            df = initialize_fps_coordinates()
            if df is not None:
                cell_list = []
                for c in df["Cell_Type"].values:
                    cell_list.append(select_cell(adata, input.cells_type_key(), c))
                fps_coordinates = [[row["x"], row["y"]] for idx, row in df.iterrows()]
                cells_indices_list = []
                for coord in fps_coordinates:
                    cells_indices_list.append(nearest_neighbors(coord, adata.obsm["X_" + input.streamline_basis()]))
                cells.set(cell_list)
                cells_indices.set(cells_indices_list)

            plt.scatter(*adata.obsm["X_" + input.streamline_basis()].T)
            if df is not None:
                for indices in cells_indices_list:
                    plt.scatter(*adata[indices[0]].obsm["X_" + input.streamline_basis()].T)
            return filter_fig(plt.gcf())

        @output
        @render.text()
        def click_info():
            return "click:\n" + json.dumps(input.initialize_searching_click(), indent=2)

        @output
        @render.text()
        def brush_info():
            return "brush:\n" + json.dumps(input.initialize_searching_brush(), indent=2)

        @output
        @render.plot()
        # @reactive.event(input.activate_streamline_plot)
        def base_streamline_plot():
            neighbors(adata, basis=input.streamline_basis(), result_prefix=input.streamline_basis())

            axes_list = streamline_plot(
                adata,
                color=input.cells_type_key().split(","),
                basis=input.streamline_basis(),
                save_show_or_return="return",
            )

            df = pd.DataFrame({
                "x": adata.obsm["X_" + input.streamline_basis()][:, 0],
                "y": adata.obsm["X_" + input.streamline_basis()][:, 1],
                "Cell_Type": adata.obs[input.cells_type_key().split(",")[0]],
                "Cell Names": adata.obs_names,
            }, index=adata.obs_names)

            coordinates_df.set(df)

            return filter_fig(plt.gcf())

        @output
        @render.table()
        def fixed_points():
            return initialize_fps_coordinates()

        @reactive.Effect
        @reactive.event(input.reset_fixed_points)
        def _():
            initialize_fps_coordinates.set(None)

        @reactive.Effect
        @reactive.event(input.activate_lap)
        def _():
            transition_graph_dict = {}
            cell_type = list(initialize_fps_coordinates()["Cell_Type"].values)
            start_cell_indices = cells_indices()
            end_cell_indices = start_cell_indices
            with ui.Progress(min=0, max=len(start_cell_indices) ** 2) as p:
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
                            ranking = rank_genes(adata, "traj_msd", output_values=True)

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
                        p.set(
                            i * len(start_cell_indices) + j,
                            message=f"Integrating step = {i * len(start_cell_indices) + j} / {len(start_cell_indices) ** 2}",
                        )
            transition_graph.set(transition_graph_dict)

            cell_type = list(initialize_fps_coordinates()["Cell_Type"].values)
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
        @render.ui
        def selectize_gene_barplot_transition():
            return ui.input_selectize(
                        "gene_barplot_transition",
                        "Specific transition to visualize:",
                        choices=list(transition_graph().keys()),
                        selected=list(transition_graph().keys())[0],
                    )

        @output
        @render.plot()
        def genes_barplot():
            if input.activate_lap() > 0:
                sns.barplot(
                    y="all",
                    x="all_values",
                    data=transition_graph()[input.gene_barplot_transition()]["ranking"][:input.top_n_genes()],
                    dodge=False,
                ).set(
                    title="Genes rank for transition: " + input.gene_barplot_transition(),
                    xlabel="ranking scores",
                    ylabel="genes",
                )

                return filter_fig(plt.gcf())

        @output
        @render.ui
        def selectize_lap_visualize_transition():
            ui_list = ui.TagList()
            for i in range(input.n_lap_visualize_transition()):
                ui_list.append(
                    ui.input_selectize(
                        "lap_visualize_transition_" + str(i),
                        "Transition " + str(i + 1) + " to visualize:",
                        choices=list(transition_graph().keys()),
                        selected=list(transition_graph().keys())[0],
                    ),
                )

            return ui_list

        @output
        @render.plot()
        def plot_lap():
            if input.activate_lap() > 0:
                paths = [getattr(input, "lap_visualize_transition_" + str(i))() for i in range(input.n_lap_visualize_transition())]
                fig, ax = plt.subplots(figsize=(5, 4))
                ax_list = streamline_plot(
                    adata,
                    basis=input.streamline_basis(),
                    save_show_or_return="return",
                    ax=ax,
                    color=input.cells_type_key().split(","),
                    frontier=True,
                )
                ax = ax_list[0]
                x, y = 0, 1

                # plot paths
                for i in range(len(paths)):
                    path = paths[i]
                    lap_dict = transition_graph()[path]["LAP_umap"]
                    for prediction, action in zip(lap_dict["prediction"], lap_dict["action"]):
                        ax.scatter(*prediction[:, [x, y]].T, c=map2color(action))
                        ax.plot(*prediction[:, [x, y]].T, c="k")

                return filter_fig(fig)

        @output
        @render.ui
        def selectize_barplot_start_genes():
            if not input.if_global_lap_time_rank():
                if initialize_fps_coordinates() is None:
                    return ui.input_text(
                        "barplot_start_genes",
                        "Starting cell type of the path: ",
                    )
                else:
                    return ui.input_selectize(
                        "barplot_start_genes",
                        "Starting cell type of the path: ",
                        choices=list(initialize_fps_coordinates()["Cell_Type"].values),
                        selected=list(initialize_fps_coordinates()["Cell_Type"].values)[0],
                    )

        @output
        @render.plot()
        def tfs_barplot():
            if input.activate_lap() > 0:
                if input.if_global_lap_time_rank():
                    develop_time_df = t_dataframe().stack().reset_index()
                    develop_time_df = develop_time_df.rename(columns={0: "integration time"})
                    develop_time_df["lineage"] = develop_time_df["level_0"] + "->" + develop_time_df["level_1"]
                else:
                    start_genes = input.barplot_start_genes()
                    develop_time_df = pd.DataFrame({"integration time": t_dataframe().loc[start_genes].T})
                    develop_time_df["lineage"] = [
                        start_genes + "->" + target for target in list(initialize_fps_coordinates()["Cell_Type"].values)
                    ]
                    develop_time_df = develop_time_df.drop(start_genes)

                ig, ax = plt.subplots(figsize=(4, 3))

                dynamo_color_dict = get_color_map_from_labels(develop_time_df["lineage"].values)

                sns.barplot(
                    y="lineage",
                    x="integration time",
                    hue="lineage",
                    data=develop_time_df,
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
        def pairwise_cell_fate_heatmap():
            if input.activate_lap() > 0:
                action_df = action_dataframe().fillna(0)
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))
                ax1 = sns.heatmap(action_df, annot=True, ax=ax1)
                t_df = t_dataframe().fillna(0)
                ax2 = sns.heatmap(t_df, annot=True, ax=ax2)
                return filter_fig(f)

        @output
        @render.ui
        def selectize_kinetic_heatmap_transition():
            return ui.input_selectize(
                "kinetic_heatmap_transition",
                "Target transition: ",
                choices=list(transition_graph().keys()),
            )

        @output
        @render.ui
        def selectize_lap_heatmap_basis():
            return ui.input_selectize(
                "lap_heatmap_basis",
                "Basis:",
                choices=[b[2:] if b.startswith("X_") else b for b in list(adata.obsm.keys())],
                selected="pca",
            )

        @output
        @render.ui
        def selectize_lap_heatmap_adj_key():
            return ui.input_selectize(
                "lap_heatmap_adj_key",
                "Adj key to locate transition matrix:",
                choices=list(adata.obsp.keys()),
                selected="cosine_transition_matrix",
            )

        @output
        @render.plot()
        def lap_kinetic_heatmap():
            if input.activate_lap() > 0:
                path = input.kinetic_heatmap_transition()
                _adata = adata.copy()
                _adata.uns["LAP_umap"] = transition_graph()[path]["LAP_umap"]
                _adata.uns["LAP_pca"] = transition_graph()[path]["LAP_pca"]
                is_human_tfs = [gene in tfs_names for gene in
                                _adata.var_names[_adata.var.use_for_transition]]
                human_genes = _adata.var_names[_adata.var.use_for_transition][is_human_tfs]
                sns.set(font_scale=0.8)
                sns_heatmap = kinetic_heatmap(
                    _adata,
                    basis=input.lap_heatmap_basis(),
                    mode="lap",
                    genes=human_genes[:input.heatmap_n_genes()],
                    project_back_to_high_dim=True,
                    save_show_or_return="return",
                    color_map="bwr",
                    transpose=False,
                    xticklabels=False,
                    yticklabels=True,
                )

                plt.setp(sns_heatmap.ax_heatmap.yaxis.get_majorticklabels())
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

        def assign_random_color(new_key: str):
            available_colors = ["#fde725", "#5ec962", "#21918c", "#3b528b", "#440154"]
            dictionary = transition_color_dict()
            available_colors = [c for c in available_colors if c not in dictionary.values()]
            random_color = random.choice(available_colors)
            dictionary[new_key] = random_color
            transition_color_dict.set(dictionary)

        @output
        @render.ui
        def selectize_known_tf_transition():
            return ui.input_selectize(
                "known_tf_transition",
                "Target transition to add known TFs:",
                choices=list(transition_graph().keys()),
            )

        @output
        @render.ui
        def selectize_reprog_mat_type():
            return ui.input_selectize(
                "reprog_mat_type",
                "Type of transition: ",
                choices=["development", "reprogramming", "transdifferentiation"],
            )

        @output
        @render.ui
        def input_reprog_mat_main_key():
            return ui.input_text("reprog_mat_main_key", "Main Key: ", value=input.known_tf_transition()),

        @output
        @render.text
        @reactive.event(input.activate_add_reprog_info)
        def add_reprog_info():
            transition = input.known_tf_transition()
            cur_transition_graph = transition_graph()

            ranking = cur_transition_graph[transition]["ranking"]
            ranking["TF"] = [i in tfs_names for i in list(ranking["all"])]
            true_tf_list = list(ranking.query("TF == True")["all"])
            all_tfs = list(ranking.query("TF == True")["all"])
            cur_transition_graph[transition][input.known_tf_key()] = input.known_tf().split(",")

            cur_transition_graph[transition][input.known_tf_rank_key()] = [
                all_tfs.index(key) if key in true_tf_list else -1 for key in
                cur_transition_graph[transition][input.known_tf_key()]
            ]

            transition_graph.set(cur_transition_graph)

            reprog_dict = reprogramming_mat_dict()
            transition_graph_dict = transition_graph()

            reprog_dict[input.reprog_mat_main_key()] = {
                "genes": transition_graph_dict[input.known_tf_transition()][input.known_tf_key()],
                "rank": transition_graph_dict[input.known_tf_transition()][input.known_tf_rank_key()],
                "type": input.reprog_mat_type(),
            }

            if input.reprog_mat_type() not in transition_color_dict().keys():
                assign_random_color(input.reprog_mat_type())

            reprogramming_mat_dict.set(reprog_dict)

            return format_dict_to_text(reprog_dict, ["genes", "type"])

        @output
        @render.ui
        def selectize_reprog_query_type():
            return ui.input_selectize(
                "reprog_query_type",
                "Query transition type to plot: ",
                choices=["development", "reprogramming", "transdifferentiation"],
            )

        @output
        @render.plot()
        def plot_priority_scores():
            if input.activate_plot_priority_scores_and_ROC() > 0:
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
        def tf_roc_curve():
            if input.activate_plot_priority_scores_and_ROC():
                all_ranks_dict = {}
                all_transitions = reprogramming_mat_dict().keys()
                for key, value in transition_graph().items():
                    if key in all_transitions:
                        ranking = transition_graph()[key]["ranking"]
                        ranking["TF"] = [i in tfs_names for i in list(ranking["all"])]
                        ranking = ranking.query("TF == True")
                        ranking["known_TF"] = [i in value[input.roc_tf_key()] for i in list(ranking["all"])]
                        all_ranks_dict[key.split("->")[0] + "_" + key.split("->")[1] + "_ranking"] = ranking

                all_ranks_df = pd.concat([rank_dict for rank_dict in all_ranks_dict.values()])

                target_ranking = all_ranks_dict[
                    list(transition_graph().keys())[0].split("->")[0] +
                    "_" +
                    list(transition_graph().keys())[0].split("->")[1] +
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