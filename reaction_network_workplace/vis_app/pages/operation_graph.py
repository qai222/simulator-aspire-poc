import os
from collections import defaultdict

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import html, get_app, Input, Output
from dash import register_page

from reaction_network.schema.lv2 import BenchTopLv2, OperationType
from reaction_network.utils import drawing_url
from reaction_network.utils import json_load
from reaction_network.visualization import STYLESHEET

app = get_app()

register_page(__name__, path='/operation_graph', description="Operations")

PAGE_ID_HEADER = "OG__"
network_path = app.title.split("-")[1].strip()  # magic

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
network = json_load(f"{THIS_DIR}/../../network_{network_path}/step03_bench_top_lv2.json")
network = BenchTopLv2(**network)
CYTO_ELEMENTS = network.to_cyto_elements()
cyto.load_extra_layouts()

COMPONENT_LEGEND = html.Div(
    [
        html.B(
            [
                html.I(className="fa fa-2x fa-square me-1", style={"color": "#800020"}),
                "Loading"
            ],
            className="me-3 d-inline-flex align-items-center"
        ),
        html.B(
            [
                html.I(className="fa fa-2x fa-square me-1", style={"color": "#808080"}),
                "Addition (Liquid)"
            ],
            className="me-3 d-inline-flex align-items-center"
        ),
        html.B(
            [
                html.I(className="fa fa-2x fa-square me-1", style={"color": "#000000"}),
                "Addition (Solid)"
            ],
            className="me-3 d-inline-flex align-items-center"
        ),
        html.B(
            [
                html.I(className="fa fa-2x fa-square me-1", style={"color": "#FF0000"}),
                "Heating"
            ],
            className="me-3 d-inline-flex align-items-center"
        ),
        html.B(
            [
                html.I(className="fa fa-2x fa-square me-1", style={"color": "#0000FF"}),
                "Purification"
            ],
            className="me-3 d-inline-flex align-items-center"
        ),
    ],
    className="mt-3 d-flex justify-content-center"
)

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
                    className="col-lg-8 px-4",
                    style={'height': 'calc(100vh - 100px)'},  # minus header bar height
                    # style={'height': '100%'},  # minus header bar height
                ),
                html.Div(
                    [
                        html.H5("Operation Graph", className="text-center mt-3"),
                        COMPONENT_LEGEND,
                        html.Div(id=DISPLAY_SELECTION_DIV_ID)
                    ],
                    className="col-lg-4 px-2",
                ),
            ]
        )

    ], style={"width": "calc(100vw - 100px)"}
)


def get_summary():
    operations_by_type = defaultdict(list)
    n_ops = 0
    for e in CYTO_ELEMENTS:
        if e['group'] == "nodes":
            n_ops += 1
            t = e['data']['data']['type']
            t = t.replace("transform_", "")
            t = " ".join(t.split("_"))
            t = t.capitalize()
            operations_by_type[t].append(e)
    unique_operation_types = sorted(operations_by_type.keys())

    row1 = html.Tr([html.Td("# of Operations"), html.Td(f"{n_ops}")])
    rows = [row1]
    for t in unique_operation_types:
        row2 = html.Tr([html.Td(t), html.Td(f"{len(operations_by_type[t])}")])
        rows.append(row2)

    table_body = [html.Tbody(rows)]

    table = dbc.Table(table_body, bordered=True,
                      hover=True,
                      responsive=True,
                      striped=True,
                      color="light"
                      )
    return table


def get_transform_table(data: dict):
    transform_id = data['id']
    precedent_ids = data['precedents']
    operation_type = data['type']

    if data['consumes'] is None:
        consumes_container = None
    else:
        consumes_container = data['consumes']['contained_by']
    if data['produces'] is None:
        produces_container = None
    else:
        produces_container = data['produces']['contained_by']

    # produces_impure = data['produces']['impure']

    row1 = html.Tr([html.Td("Operation ID"), html.Td(transform_id)])
    row2 = html.Tr([html.Td("Type"), html.Td(operation_type)])
    row3 = html.Tr([html.Td("# of Precedents"), html.Td(f"{len(precedent_ids)}")])
    row4 = html.Tr([html.Td("can be realized by"), html.Td(f"{' OR '.join(data['can_be_realized_by'])}")])

    rows = [row1, row2, row3, row4]

    if operation_type in [OperationType.OPERATION_LOADING, OperationType.OPERATION_RELOADING,
                          OperationType.OPERATION_ADDITION_SOLID, OperationType.OPERATION_ADDITION_LIQUID]:
        produces_compounds = data['produces']['compounds']
        assert len(produces_compounds) == 1
        pc = produces_compounds[0]
        pc_smi = pc['compound_lv0']['smiles']
        pc_amount_string = "{:.3f} ({})".format(pc['amount'], pc['amount_unit'])
        url = drawing_url(pc_smi, size=80)
        row4 = html.Tr([html.Td("Transferring Compound"), html.Td(html.Img(src=url))])
        row5 = html.Tr([html.Td("Transferring Amount"), html.Td(pc_amount_string)])
        row6 = html.Tr([html.Td("From Container"), html.Td(consumes_container)])
        row7 = html.Tr([html.Td("To Container"), html.Td(produces_container)])
        rows += [row4, row5, row6, row7]

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
        ]
    else:
        data = data_list[-1]['data']
        children = [
            html.Hr(),
            html.H5("Operation Details", className="text-center mt-3"),
            get_transform_table(data),
        ]
    return children
