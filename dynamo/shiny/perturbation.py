import matplotlib.pyplot as plt
import shiny.experimental as x
from shiny import App, reactive, render, ui

from ..plot import streamline_plot
from ..prediction import perturbation
from ..sample_data import hematopoiesis


def perturbation_web_app():
    app_ui = x.ui.page_sidebar(
        x.ui.sidebar(
            x.ui.accordion(
                x.ui.accordion_panel(
                    "Perturbation",
                    ui.input_text("selected_genes", "Genes", placeholder="Enter genes"),
                    ui.input_text("emb_basis", "basis", placeholder="Enter basis"),
                    ui.input_text("expression", "expression", placeholder="Enter expression"),
                ),
                x.ui.accordion_panel(
                    "Streamline Plot",
                    ui.input_text("color", "color", placeholder="Enter color"),
                    ui.input_text("streamline_basis", "output basis", placeholder="Enter basis"),
                ),
            ),
        ),
        ui.div(
            ui.input_action_button(
                "run", "Run simulation", class_="btn-primary"
            )
        ),
        x.ui.output_plot("perturbation_plot"),
    )

    def server(input, output, session):
        @output
        @render.plot()
        @reactive.event(input.run)
        def perturbation_plot():

            if input.run() > 0:
                adata = hematopoiesis()

                selected_genes = input.selected_genes().split(",")
                expression = [int(txt) for txt in input.expression().split(",")]

                perturbation(adata, selected_genes, expression, emb_basis=input.emb_basis())
                color = input.color().split(",")

                return streamline_plot(adata, color=color, basis=input.streamline_basis(), save_show_or_return="return")

            return plt.gcf()

    app = App(app_ui, server, debug=True)
    app.run()