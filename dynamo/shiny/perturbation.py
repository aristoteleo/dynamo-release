import matplotlib.pyplot as plt
import shiny.experimental as x
from shiny import App, render, ui

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
        x.ui.output_plot("perturbation_plot"),
    )

    def server(input, output, session):
        @output
        @render.plot()
        def perturbation_plot():
            adata = hematopoiesis()

            color = input.color().split(",")
            selected_genes = input.selected_genes().split(",")
            expression = [int(txt) for txt in input.expression().split(",")]

            perturbation(adata, selected_genes, expression, emb_basis=input.emb_basis())

            return streamline_plot(adata, color=color, basis=input.streamline_basis(), save_show_or_return="return")

    app = App(app_ui, server, debug=True)
    app.run()