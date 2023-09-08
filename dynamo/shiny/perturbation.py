import matplotlib.pyplot as plt
from shiny import App, render, ui

from ..plot import streamline_plot
from ..prediction import perturbation
from ..sample_data import hematopoiesis


def perturbation_web_app():
    app_ui = ui.page_fluid(
        ui.layout_sidebar(
            ui.panel_sidebar(
                ui.input_text("selected_genes", "Genes", placeholder="Enter genes"),
                ui.input_text("emb_basis", "basis", placeholder="Enter basis"),
                ui.input_text("color", "color", placeholder="Enter color"),
                ui.input_text("streamline_basis", "output basis", placeholder="Enter basis"),
            ),
            ui.panel_main(
                ui.output_plot("perturbation_plot"),
            ),
        ),
    )

    def server(input, output, session):
        @output
        @render.plot()
        def perturbation_plot():
            adata = hematopoiesis()

            color = input.color().split(",")
            selected_genes = input.selected_genes().split(",")

            perturbation(adata, selected_genes, emb_basis=input.emb_basis())

            return streamline_plot(adata, color=color, basis=input.streamline_basis(), save_show_or_return="return")

    app = App(app_ui, server, debug=True)
    app.run()