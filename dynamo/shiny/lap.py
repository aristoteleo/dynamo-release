import matplotlib.pyplot as plt
import numpy as np
import shiny.experimental as x
from shiny import App, reactive, render, ui

from .utils import filter_fig
from ..plot import streamline_plot
from ..tools import select_cell


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
        x.ui.output_plot("base_streamline_plot"),
        x.ui.output_plot("initialize_searching"),
    )

    def server(input, output, session):
        adata = input_adata.copy()
        cells = reactive.Value[list[np.ndarray]]()

        @output
        @render.plot()
        @reactive.event(input.initialize)
        def initialize_searching():
            cells_names = input.cells_names().split(",")
            cell_list = []
            for c in cells_names:
                cell_list.append(select_cell(adata, input.cells_type_key(), c))

            cells.set(cell_list)
            plt.scatter(*adata.obsm["X_" + input.streamline_basis()].T)

        @output
        @render.plot()
        @reactive.event(input.activate_streamline_plot)
        def base_streamline_plot():
            color = input.cells_type_key().split(",")

            axes_list = streamline_plot(adata, color=color, basis=input.streamline_basis(), save_show_or_return="return")

            return filter_fig(plt.gcf())


    app = App(app_ui, server, debug=True)
    app.run()