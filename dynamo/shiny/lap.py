import matplotlib.pyplot as plt
import shiny.experimental as x
from shiny import App, reactive, render, ui

from .utils import filter_fig
from ..plot import streamline_plot


def lap_web_app(input_adata):
    app_ui = x.ui.page_sidebar(
        x.ui.sidebar(
            x.ui.accordion(
                x.ui.accordion_panel(
                    "Streamline Plot",
                    ui.input_text("color", "color", placeholder="Enter color"),
                    ui.input_text("streamline_basis", "output basis", placeholder="Enter basis"),
                ),
            ),
        ),
        ui.div(
            ui.input_action_button(
                "activate_streamline_plot", "Run streamline plot", class_="btn-primary"
            )
        ),
        x.ui.output_plot("base_streamline_plot"),
    )

    def server(input, output, session):
        adata = input_adata.copy()

        @output
        @render.plot()
        @reactive.event(input.activate_streamline_plot)
        def base_streamline_plot():
            color = input.color().split(",")

            axes_list = streamline_plot(adata, color=color, basis=input.streamline_basis(), save_show_or_return="return")

            return filter_fig(plt.gcf())


    app = App(app_ui, server, debug=True)
    app.run()