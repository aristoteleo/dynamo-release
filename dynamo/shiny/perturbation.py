import matplotlib.pyplot as plt
import pandas as pd
from anndata import AnnData
from pathlib import Path

from .utils import filter_fig
from ..plot import streamline_plot
from ..prediction import perturbation


css_path = Path(__file__).parent / "styles.css"


def perturbation_web_app(input_adata: AnnData):
    """The Shiny web application of the in silico perturbation. The process is equivalent to this tutorial:
    https://dynamo-release.readthedocs.io/en/latest/notebooks/perturbation_tutorial/perturbation_tutorial.html

    Args:
        input_adata: the processed anndata object to perform in silico perturbation.
    """
    try:
        import shiny.experimental as x
        from htmltools import HTML, div
        from shiny import App, Inputs, Outputs, reactive, Session, render, ui
    except ImportError:
        raise ImportError("Please install shiny and htmltools before running the web application!")

    app_ui = x.ui.page_sidebar(
        x.ui.sidebar(
            ui.include_css(css_path),
            x.ui.accordion(
                x.ui.accordion_panel(
                    div("Perturbation Setting", class_="bold-subtitle"),
                    ui.input_slider("n_genes", "Number of genes to perturb:", min=1, max=5, value=1),
                    ui.output_ui("selectize_genes"),
                    ui.input_action_button(
                        "activate_perturbation", "Run perturbation", class_="btn-primary"
                    ),
                    value="Perturbation",
                ),
                x.ui.accordion_panel(
                    div("Streamline Plot Setting", class_="bold-subtitle"),
                    ui.input_slider("n_colors", "Number of observations:", min=1, max=5, value=1),
                    ui.output_ui("selectize_color"),
                    ui.output_ui("selectize_basis"),
                    value="Streamline Plot Setting",
                ),
                open=True,
            ),
            width=500,
        ),
        ui.div(
            div("in silico perturbation", class_="bold-title"),
            div(HTML("<br><br>")),
            div("Perturbation function in Dynamo can be used to either upregulating or suppressing a single or "
                "multiple genes in a particular cell or across all cells to perform in silico genetic perturbation. "
                "Dynamo first calculates the perturbation velocity vector from the input expression value and "
                "the analytical Jacobian from our vector field function Because Jacobian encodes the instantaneous "
                "changes of velocity of any genes after increasing any other gene, the output vector will produce the "
                "perturbation effect vector after propagating the genetic perturbation through the gene regulatory "
                "network. Then Dynamo projects the perturbation vector to low dimensional space.",
                class_="explanation"),
            div(HTML("<br><br>")),
            x.ui.card(
                div("Streamline Plot", class_="bold-subtitle"),
                x.ui.output_plot("base_plot"),
            ),
            x.ui.card(
                div("Streamline Plot After Perturbation", class_="bold-subtitle"),
                x.ui.output_plot("perturbation_plot"),
            ),
        ),
    )

    def server(input: Inputs, output: Outputs, session: Session):
        adata = input_adata.copy()

        @output
        @render.ui
        def selectize_color():
            ui_list = ui.TagList()
            for i in range(input.n_colors()):
                ui_list.append(
                    ui.input_selectize(
                        "base_color_" + str(i),
                        "Color key " + str(i + 1) + " :",
                        choices=list(adata.obs.keys()) + list(adata.var_names),
                        selected="cell_type",
                    ),
                )

            return ui_list

        @output
        @render.ui
        def selectize_basis():
            return ui.input_selectize(
                        "streamline_basis",
                        "The perturbation output as the basis of plot: ",
                        choices=[b[2:] if b.startswith("X_") else b for b in list(adata.obsm.keys())],
                        selected="umap",
                    )

        @output
        @render.ui
        def selectize_genes():
            ui_list = ui.TagList()
            for i in range(input.n_genes()):
                ui_list.extend(
                    (
                        ui.input_selectize(
                            "target_gene_" + str(i),
                            "Genes " + str(i + 1) + " to perform perturbation:",
                            choices=list(adata.var_names),
                        ),
                        ui.input_slider(
                            "expression_" + str(i),
                            "Expression value to encode the genetic perturbation: ",
                            min=-200, max=200, value=-100,
                        ),
                    ),

                )
            return ui_list

        @output
        @render.plot()
        def base_plot():
            color = [getattr(input, "base_color_" + str(i))() for i in range(input.n_colors())]

            axes_list = streamline_plot(adata, color=color, basis=input.streamline_basis(),
                                        save_show_or_return="return")

            return filter_fig(plt.gcf())

        @reactive.Effect
        @reactive.event(input.activate_perturbation)
        def activate_perturbation():
            selected_genes = [getattr(input, "target_gene_" + str(i))() for i in range(input.n_genes())]
            expression = [getattr(input, "expression_" + str(i))() for i in range(input.n_genes())]

            perturbation(adata, selected_genes, expression, emb_basis=input.streamline_basis())

        @output
        @render.plot()
        def perturbation_plot():
            if input.activate_perturbation() > 0:
                color = [getattr(input, "base_color_" + str(i))() for i in range(input.n_colors())]
                axes_list = streamline_plot(adata, color=color, basis=input.streamline_basis() + "_perturbation", save_show_or_return="return")

                return filter_fig(plt.gcf())

    app = App(app_ui, server, debug=True)
    app.run()
