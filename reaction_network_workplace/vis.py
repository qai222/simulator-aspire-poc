from __future__ import annotations

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_renderjson
from dash import Input, Output, no_update, Dash, html

from reaction_network.utils import json_load
from reaction_network.visualization import JsonTheme
from reaction_network.schema import NetworkLv1

"""
define a reaction network

LV0. given the routes selection seed, scraper_input, and scraper_output
LV1. given the target product amounts, and expected yields
"""

network = json_load("network_lv1/network_lv1.json")
network = NetworkLv1(**network)

CYTO_ELEMENTS = network.to_cyto_elements()

cyto.load_extra_layouts()
app = Dash(
    __name__,
    title="Synthesis Campaign",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

STYLESHEET = [
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'unbundled-bezier',
            'taxi-direction': 'vertical',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': 'black',
            "opacity": "0.9",
            "line-color": "black",
            # "width": "mapData(weight, 0, 1, 1, 8)",
            "overlay-padding": "3px"
        }
    },
    {
        'selector': '.reaction',
        'style': {
            'width': 500,
            'height': 200,
            'shape': 'rectangle',
            'background-fit': 'contain',
            'background-image': 'data(url)',
            "border-width": "6px",
            "border-color": "red",
            "border-opacity": "1.0",
            "background-color": "white",
        }
    },
    {
        'selector': '.compound',
        'style': {
            'width': 200,
            'height': 200,
            'shape': 'circle',
            'background-fit': 'contain',
            'background-image': 'data(url)',
            "border-width": "6px",
            "border-color": "#AAD8FF",
            "border-opacity": "1.0",
            "background-color": "white",
            # "content": 'data(label)',
            # "text-outline-color": "#77828C"
        }
    },
    {
        'selector': ':selected',
        'style': {
            'z-index': 1000,
            'background-color': 'SteelBlue',
            'line-color': 'SteelBlue',
        }
    },
]

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
