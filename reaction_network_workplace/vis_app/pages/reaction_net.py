import os

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import html, get_app, Input, Output
from dash import register_page

from reaction_network.schema.lv2 import NetworkLv1
from reaction_network.utils import drawing_url
from reaction_network.utils import json_load
from reaction_network.visualization import STYLESHEET

register_page(__name__, path='/reaction_network', description="Reactions")

PAGE_ID_HEADER = "RN__"

app = get_app()
network_path = app.title.split("-")[1].strip()  # magic
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
network = json_load(f"{THIS_DIR}/../../network_{network_path}/step02_network_lv1.json")
network = NetworkLv1(**network)
CYTO_ELEMENTS = network.to_cyto_elements(size=500)
cyto.load_extra_layouts()

CYTO_ID = PAGE_ID_HEADER + "CYTO"
DISPLAY_SELECTION_DIV_ID = PAGE_ID_HEADER + "DISPLAY_SELECTION"

layout = html.Div(
    [
        dbc.Row(
            [
                html.Div(
                    id=f"{PAGE_ID_HEADER}CYTO_DIV",
                    children=
                    [
                        cyto.Cytoscape(
                            id=CYTO_ID,
                            layout={
                                'name': 'dagre',
                                'nodeDimensionsIncludeLabels': True,
                                'animate': True,
                                'animationDuration': 1000,
                                'rankDir': 'LR',
                                'align': 'UL',
                            },
                            style={
                                'width': '100%',
                                'height': '100%'
                            },
                            className="border-primary border",
                            elements=CYTO_ELEMENTS,
                            stylesheet=STYLESHEET,
                            responsive=True,
                        )
                    ],
                    className="col-lg-12 px-4",
                    style={'height': 'calc(100vh - 100px)'},  # minus header bar height
                    # style={'height': '100%'},  # minus header bar height
                ),
                html.Div(
                    [
                        html.H5("Reaction Network", className="text-center mt-3"),
                        html.Div(id=DISPLAY_SELECTION_DIV_ID)
                    ],
                    className="col-lg-4 px-2 overflow-auto",
                    style={'height': 'calc(100vh - 100px)'},  # minus header bar height
                ),
            ]
        )

    ], style={"width": "calc(100vw - 100px)"}
)


def get_summary():
    rows = []
    for k, v in network.summary.items():
        row = html.Tr([html.Td(k), html.Td(str(v))])
        rows.append(row)

    table_body = [html.Tbody(rows)]

    table = dbc.Table(table_body, bordered=True,
                      hover=True,
                      responsive=True,
                      striped=True,
                      color="light"
                      )
    return table


def get_target_table():
    rows = []
    for smi in network.network_lv0.target_smis:
        compound = network.compound_dict[smi]
        row = html.Tr(
            [
                html.Td(html.Img(src=drawing_url(compound.compound_lv0.smiles, size=100))),
                html.Td("{:.4f} GRAM".format(compound.mass))
            ]
        )
        rows.append(row)

    table_body = [html.Tbody(rows)]

    table = dbc.Table(table_body, bordered=True,
                      hover=True,
                      responsive=True,
                      striped=True,
                      color="light"
                      )
    return table


def get_compound_table(data: dict):
    amount_string = "{:.3f} ({})".format(data['amount'], data['amount_unit'])
    moles = data['moles']
    smiles = data['compound_lv0']['smiles']
    row1 = html.Tr([html.Td("Compound Structure"), html.Td(html.Img(src=drawing_url(smiles)))])
    row2 = html.Tr([html.Td("Amount"), html.Td(amount_string)])
    row3 = html.Tr([html.Td("Moles"), html.Td("{:.6f}".format(moles))])

    rows = [row1, row2, row3]
    for k, v in data['compound_lv0'].items():
        row = html.Tr([html.Td(k), html.Td(v)])
        rows.append(row)

    table_body = [html.Tbody(rows)]

    table = dbc.Table(table_body, bordered=True,
                      hover=True,
                      responsive=True,
                      striped=True,
                      color="light"
                      )
    return table


def get_reaction_table(data: dict):
    row1 = html.Tr(
        [
            html.Td("Reaction"),
            html.Td(html.Img(src=drawing_url(data['reaction_lv0']['reaction_smiles'], size=200))),
        ]
    )
    row2 = html.Tr(
        [
            html.Td("Reaction smiles"),
            html.Td(data['reaction_lv0']['reaction_smiles']),
        ]
    )
    row3 = html.Tr(
        [
            html.Td("Expected yield"),
            html.Td("{:.2%}".format(data['expected_yield'])),
        ]
    )
    row4 = html.Tr(
        [
            html.Td("Batch extent"),
            html.Td(data['batch_size']),
        ]
    )
    temperature = data['reaction_lv0']['temperature']
    row5 = html.Tr(
        [
            html.Td("Temperature"),
            html.Td("{:.1f} °C".format(temperature)),
        ]
    )

    rows = [row1, row2]  # TODO tune their sizes...
    rows = [row3, row4, row5]

    for i, r in enumerate(data['reactants']):
        amount_string = "{:.3f} ({})".format(r['amount'], r['amount_unit'])
        smiles = r['compound_lv0']['smiles']
        k1 = html.Td(f"Reactant {i} structure")
        v1 = html.Td(html.Img(src=drawing_url(smiles)))
        k2 = html.Td(f"Reactant {i} smiles")
        v2 = html.Td(smiles)
        k3 = html.Td(f"Reactant {i} amount")
        v3 = html.Td(amount_string)
        compound_rows = [
            html.Tr([k1, v1]),
            html.Tr([k2, v2]),
            html.Tr([k3, v3]),
        ]
        rows += compound_rows

    for i, r in enumerate(data['reagents']):
        amount_string = "{:.3f} ({})".format(r['amount'], r['amount_unit'])
        smiles = r['compound_lv0']['smiles']
        k1 = html.Td(f"Reagent {i} structure")
        v1 = html.Td(html.Img(src=drawing_url(smiles)))
        k2 = html.Td(f"Reagent {i} smiles")
        v2 = html.Td(smiles)
        k3 = html.Td(f"Reagent {i} amount")
        v3 = html.Td(amount_string)
        compound_rows = [
            html.Tr([k1, v1]),
            html.Tr([k2, v2]),
            html.Tr([k3, v3]),
        ]
        rows += compound_rows

    r = data['product']
    amount_string = "{:.3f} ({})".format(r['amount'], r['amount_unit'])
    smiles = r['compound_lv0']['smiles']
    k1 = html.Td(f"Product structure")
    v1 = html.Td(html.Img(src=drawing_url(smiles)))
    k2 = html.Td(f"Product smiles")
    v2 = html.Td(smiles)
    k3 = html.Td("Product Amount")
    v3 = html.Td(amount_string)
    compound_rows = [
        html.Tr([k1, v1]),
        html.Tr([k2, v2]),
        html.Tr([k3, v3]),
    ]
    rows += compound_rows

    table_body = [html.Tbody(rows)]

    table = dbc.Table(table_body, bordered=True,
                      hover=True,
                      responsive=True,
                      striped=True,
                      color="light"
                      )
    return table


@app.callback(Output(DISPLAY_SELECTION_DIV_ID, 'children'),
              Input(CYTO_ID, 'selectedNodeData'))
def display_selected_node(data_list):
    if data_list is None or len(data_list) == 0:
        children = [
            html.Hr(),
            html.H5("Summary", className="text-center mt-3"),
            get_summary(),
            html.H5("Targets", className="text-center mt-3"),
            get_target_table(),
        ]
    else:
        d = data_list[-1]
        smi = d['id']
        data = data_list[-1]['data']
        if ">>" not in smi:
            content = get_compound_table(data)
        else:
            content = get_reaction_table(data)
        children = [
            html.Hr(),
            html.H5("Selection Details", className="text-center mt-3"),
            content
        ]
    return children
