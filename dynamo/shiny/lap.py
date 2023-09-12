import matplotlib.pyplot as plt
import numpy as np
import shiny.experimental as x
from shiny import App, reactive, render, ui

from .utils import filter_fig
from ..prediction import GeneTrajectory, least_action, least_action_path
from ..plot import streamline_plot
from ..plot.utils import map2color
from ..tools import neighbors
from ..tools.utils import nearest_neighbors, select_cell
from ..vectorfield import rank_genes


def lap_web_app(input_adata):
    app_ui = x.ui.page_sidebar(
        x.ui.sidebar(
            x.ui.accordion(
                x.ui.accordion_panel(
                    "Streamline Plot",
                    ui.input_text("cells_type_key", "cells type key", placeholder="cells type key"),
                    ui.input_text("streamline_basis", "output basis", placeholder="Enter basis"),
                ),
            ),
            x.ui.accordion(
                x.ui.accordion_panel(
                    "Initialization",
                    ui.input_text("cells_names", "cells names", placeholder="Enter names of cell"),
                    ui.input_text(
                        "fps_coordinates",
                        "fixed points coordinates",
                        placeholder="Enter the coordinates of fixed point."
                    ),
                ),
            ),
            x.ui.accordion(
                x.ui.accordion_panel(
                    "Visualize LAP",
                    ui.input_text("visualize_keys", "keys", placeholder="Enter keys"),
                ),
            ),
        ),
        ui.div(
            ui.input_action_button(
                "activate_streamline_plot", "Run streamline plot", class_="btn-primary"
            )
        ),
        ui.div(
            ui.input_action_button(
                "initialize", "Initialize searching", class_="btn-primary"
            )
        ),
        ui.div(
            ui.input_action_button(
                "activate_lap", "Run LAP analyses", class_="btn-primary"
            )
        ),
        ui.div(
            ui.input_action_button(
                "activate_visualize_lap", "Visualize LAP", class_="btn-primary"
            )
        ),
        x.ui.output_plot("base_streamline_plot"),
        x.ui.output_plot("initialize_searching"),
    )

    def server(input, output, session):
        adata = input_adata.copy()
        cells = reactive.Value[list[np.ndarray]]()
        cells_indices = reactive.Value[list[list[float]]]()
        transition_graph = reactive.Value[dict]()

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
            plt.scatter(*adata.obsm["X_" + input.streamline_basis()].T)
            for indices in cells_indices_list:
                plt.scatter(*adata[indices[0]].obsm["X_" + input.streamline_basis()].T)

        @output
        @render.plot()
        @reactive.event(input.activate_streamline_plot)
        def base_streamline_plot():
            neighbors(adata, basis=input.streamline_basis(), result_prefix=input.streamline_basis())

            color = input.cells_type_key().split(",")

            axes_list = streamline_plot(adata, color=color, basis=input.streamline_basis(), save_show_or_return="return")

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
                        least_action(adata, basis="umap")
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
                lap_dict = transition_graph[path]["LAP_umap"]
                for prediction, action in zip(lap_dict["prediction"], lap_dict["action"]):
                    ax.scatter(*prediction[:, [x, y]].T, c=map2color(action))
                    ax.plot(*prediction[:, [x, y]].T, c="k")

            return filter_fig(fig)

    app = App(app_ui, server, debug=True)
    app.run()