from __future__ import annotations

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_renderjson
from dash import Input, Output, no_update, Dash, html

from reaction_network.utils import json_load
from reaction_network.visualization import JsonTheme, STYLESHEET
from reaction_network.schema.lv2 import BenchTopLv2, NetworkLv1

"""
define a reaction network

LV0. given the routes selection seed, scraper_input, and scraper_output
LV1. given the target product amounts, and expected yields
"""

# network = json_load("network_lv1/network_lv1.json")
# network = NetworkLv1(**network)
#
# CYTO_ELEMENTS = network.to_cyto_elements()

network = json_load("step03_bench_top_lv2.json")
network = BenchTopLv2(**network)
CYTO_ELEMENTS = network.to_cyto_elements()


cyto.load_extra_layouts()
app = Dash(
    __name__,
    title="Synthesis Campaign",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)



app.layout = html.Div(
    [
        dbc.Row(
            [
                html.Div(
                    id="CYTO_DIV",
                    children=
                    [
                        html.H5("Reaction Network", className="text-center mt-3"),
                        html.Hr(),
                        cyto.Cytoscape(
                            id='CYTO',
                            layout={
                                'name': 'dagre',
                                'nodeDimensionsIncludeLabels': True,
                                'animate': True,
                                'animationDuration': 1000,
                                'align': 'UL',
                            },
                            style={
                                'width': '100%',
                                'height': 'calc(100vh - 100px)'
                            },
                            className="border-primary border",
                            elements=CYTO_ELEMENTS,
                            stylesheet=STYLESHEET,
                            responsive=True,
                        )
                    ],
                    className="col-lg-8 px-4",
                    style={'height': 'calc(100vh - 100px)'},  # minus header bar height
                    # style={'height': '100%'},  # minus header bar height
                ),
                html.Div(
                    [
                        html.H5("Campaign Summary", className="text-center mt-3"),
                        dash_renderjson.DashRenderjson(max_depth=-1, theme=JsonTheme, data=network.summary,
                                                       invert_theme=True, ),
                        html.Hr(),
                        html.H5("Selected Node", className="text-center mt-3"),
                        dash_renderjson.DashRenderjson(id='selected-node-json', max_depth=-1, theme=JsonTheme,
                                                       invert_theme=True, ),
                        html.Hr(),
                    ],
                    className="col-lg-4 px-2",
                ),
            ]
        )

    ], style={"width": "calc(100vw - 100px)"}
)


@app.callback(Output('selected-node-json', 'data'),
              Input('CYTO', 'selectedNodeData'))
def display_selected_node(data_list):
    if data_list is None or len(data_list) == 0:
        return no_update

    return data_list[-1]['data']


if __name__ == '__main__':
    app.run(debug=True)
