import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shiny.experimental as x
from shiny import App, reactive, render, ui

from .utils import filter_fig
from ..prediction import GeneTrajectory, least_action, least_action_path
from ..plot import kinetic_heatmap, streamline_plot
from ..plot.utils import map2color
from ..tools import neighbors
from ..tools.utils import nearest_neighbors, select_cell
from ..vectorfield import rank_genes


def lap_web_app(input_adata, tfs_data):
    app_ui = x.ui.page_sidebar(
        x.ui.sidebar(
            x.ui.accordion(
                x.ui.accordion_panel(
                    "Streamline Plot",
                    ui.input_text("cells_type_key", "cells type key", placeholder="cells type key"),
                    ui.input_text("streamline_basis", "output basis", placeholder="Enter basis"),
                ),
                x.ui.accordion_panel(
                    "Initialization",
                    ui.input_text("cells_names", "cells names", placeholder="Enter names of cell"),
                    ui.input_text(
                        "fps_coordinates",
                        "fixed points coordinates",
                        placeholder="Enter the coordinates of fixed point."
                    ),
                ),
                x.ui.accordion_panel(
                    "Visualize LAP",
                    ui.input_text("visualize_keys", "keys", placeholder="Enter keys"),
                ),
                x.ui.accordion_panel(
                    "TFs barplot",
                    ui.input_text("cell_type_colormap", "cell type colormap", placeholder="Enter Color Dict"),
                ),
                x.ui.accordion_panel(
                    "LAP Kinetic Heatmap",
                    ui.input_text("lap_init_cells", "Init Cells", placeholder="Enter init cells"),
                    ui.input_text("lap_end_cells", "End Cells", placeholder="Enter end cells"),
                    ui.input_text("lap_basis", "Basis", value="pca", placeholder="Enter basis"),
                    ui.input_text("lap_adj_key", "Adj Key", value="cosine_transition_matrix", placeholder="Enter adj key"),
                ),
            ),
        ),
        ui.div(
            ui.input_action_button(
                "activate_streamline_plot", "Run streamline plot", class_="btn-primary"
            ),
            ui.input_action_button(
                "initialize", "Initialize searching", class_="btn-primary"
            ),
            ui.input_action_button(
                "activate_lap", "Run LAP analyses", class_="btn-primary"
            ),
            ui.input_action_button(
                "activate_visualize_lap", "Visualize LAP", class_="btn-primary"
            ),
            ui.input_action_button(
                "activate_prepare_tfs", "Prepare TFs", class_="btn-primary"
            ),
            ui.input_action_button(
                "activate_tfs_barplot", "TFs barplot", class_="btn-primary"
            ),
            ui.input_action_button(
                "activate_pairwise_cell_fate_heatmap", "Pairwise cell fate heatmap", class_="btn-primary"
            ),
            ui.input_action_button(
                "activate_lap_kinetic_heatmap", "LAP kinetic heatmap", class_="btn-primary"
            ),
        ),
        ui.div(
            x.ui.output_plot("base_streamline_plot"),
            x.ui.output_plot("initialize_searching"),
            x.ui.output_plot("plot_lap"),
            x.ui.output_plot("tfs_barplot"),
            x.ui.output_plot("pairwise_cell_fate_heatmap"),
            x.ui.output_plot("lap_kinetic_heatmap"),
        ),
    )

    def server(input, output, session):
        adata = input_adata.copy()
        tfs_names = list(tfs_data["Symbol"])
        cells = reactive.Value[list[np.ndarray]]()
        cells_indices = reactive.Value[list[list[float]]]()
        transition_graph = reactive.Value[dict]()
        t_dataframe = reactive.Value[pd.DataFrame]()
        action_dataframe = reactive.Value[pd.DataFrame]()

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
            colors = input.cell_type_colormap().split(",")
            dynamo_color_dict = {}
            for i in range(len(develop_time_df["lineage"])):
                dynamo_color_dict[develop_time_df["lineage"][i]] = colors[i]

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


    app = App(app_ui, server, debug=True)
    app.run()