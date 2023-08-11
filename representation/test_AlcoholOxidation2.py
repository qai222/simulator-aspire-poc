import json

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_renderjson
from dash import Dash, html, Input, Output

from representation.base import *

JsonTheme = {
    "scheme": "monokai",
    "author": "wimer hazenberg (http://www.monokai.nl)",
    "base00": "#272822",
    "base01": "#383830",
    "base02": "#49483e",
    "base03": "#75715e",
    "base04": "#a59f85",
    "base05": "#f8f8f2",
    "base06": "#f5f4f1",
    "base07": "#f9f8f5",
    "base08": "#f92672",
    "base09": "#fd971f",
    "base0A": "#f4bf75",
    "base0B": "#a6e22e",
    "base0C": "#a1efe4",
    "base0D": "#66d9ef",
    "base0E": "#ae81ff",
    "base0F": "#cc6633",
}


def create_world():
    HardwareClass_MRV = HardwareClass(name="MRV", description="vials used for reactions")
    # HardwareClass_HRV = HardwareClass(name="HRV", description="vials typically used in liquid dispensing")
    HardwareClass_LH = HardwareClass(name="Liquid Handler", description="machine for liquid transfer")
    HardwareClass_SH = HardwareClass(name="Solid Handler", description="machine for solid transfer")

    class Vial(HardwareUnit):
        volume_capacity: float

        has_stirring_bar: bool = True

    mrv = Vial(hardware_class=HardwareClass_MRV, volume_capacity=5.0, has_stirring_bar=True,
               made_of=[ChemicalIdentifier(value="GLASS"), ])

    lh = HardwareUnit(hardware_class=HardwareClass_LH, made_of=[ChemicalIdentifier(value="PLASTIC")])

    sh = HardwareUnit(hardware_class=HardwareClass_SH, made_of=[ChemicalIdentifier(value="STEEL")])

    compound_r1_ci = ChemicalIdentifier(value="triazolybenzenemethanol")
    compound_r1 = Compound(chemical_identifier=compound_r1_ci, amount=1.0, amount_unit="eq", )

    compound_r2_ci = ChemicalIdentifier(value="TEMPO")
    compound_r2 = Compound(chemical_identifier=compound_r2_ci, amount=0.2, amount_unit="eq")

    compound_r3_ci = ChemicalIdentifier(value="BAIB")
    compound_r3 = Compound(chemical_identifier=compound_r3_ci, amount=2.25, amount_unit="eq")

    compound_s1_ci = ChemicalIdentifier(value="DCM")
    compound_s1 = Compound(chemical_identifier=compound_s1_ci, amount=5.0, amount_unit="mL")

    compound_s2_ci = ChemicalIdentifier(value="Water")
    compound_s2 = Compound(chemical_identifier=compound_s2_ci, amount=5.0, amount_unit="mL")

    action_p1 = Action(
        description="Add solid R1 to MRV", precedents=[],
        inputs=[compound_r1_ci], outputs=[compound_r1],
        uses_hardware_unit=[mrv, sh],
    )
    action_p2 = Action(
        description="Add solid R2 to MRV", precedents=[],
        inputs=[compound_r2_ci], outputs=[compound_r2],
        uses_hardware_unit=[mrv, sh],
    )

    action_p3 = Action(
        description="Add solid R3 to MRV", precedents=[],
        inputs=[compound_r3_ci], outputs=[compound_r3],
        uses_hardware_unit=[mrv, sh],
    )

    action_p4 = Action(
        description="Add S1 to MRV", precedents=[action_p3, action_p2, action_p1],
        inputs=[compound_s1_ci], outputs=[compound_s1],
        uses_hardware_unit=[mrv, lh],
    )

    action_p5 = Action(
        description="Add S2 to MRV", precedents=[action_p4],
        inputs=[compound_s2_ci], outputs=[compound_s2],
        uses_hardware_unit=[mrv, lh],
    )

    action_p6 = Action(
        description="react for 24 h", precedents=[action_p5],
        inputs=[compound_s1_ci, compound_s2_ci, compound_r1_ci, compound_r2_ci, compound_r3_ci], outputs=[],
        uses_hardware_unit=[mrv],
    )
    return WORLD


create_world()


def dump_world(w: nx.MultiDiGraph, fn: str = "world.json"):
    nodes = []
    for node, d in w.nodes(data=True):
        nodes.append(d['individual'].model_dump())
    edges = []
    for u, v, k, d in w.edges(data=True, keys=True):
        edges.append([u, v, k, d['object_property'].model_dump()])
    with open(fn, "w") as f:
        json.dump({"nodes": nodes, "edges": edges}, f)


dump_world(WORLD)


def individual_to_cyto_element(individual: Individual, index: int):
    if isinstance(individual, (ChemicalIdentifier)):
        node_label = individual.value
    elif isinstance(individual, HardwareClass):
        node_label = individual.name
    else:
        node_label = f"{individual.__class__.__name__}-{index}"

    ele_attrs = individual.model_dump()
    ele_attrs.update({"individual class": individual.__class__.__name__})

    data = {
        "id": individual.identifier,
        "label": node_label,
        "ele_attrs": ele_attrs,
    }
    cyto_element = dict(
        group="nodes",
        data=data,
        selected=False,
        selectable=True,
        grabbable=True,
        locked=False,
        classes=individual.__class__.__name__,
    )
    return cyto_element


def world_to_cyto():
    individuals = find_individuals()
    cyto_nodes = [individual_to_cyto_element(i, index=ind) for ind, i in enumerate(individuals)]
    cyto_edges = []

    def get_edge_color(pred: str):
        if pred == "inputs":
            return "blue"
        # if pred == "precedents":
        #     return "red"
        return None

    for u, v, k, d in WORLD.edges(data=True, keys=True):
        op = d['object_property']
        op: ObjectProperty
        cyto_edge = dict(
            group="edges",
            data={
                "id": op.subject_identifier + " | " + op.object_identifier,
                "source": op.subject_identifier,
                "target": op.object_identifier,
                "predicate": op.predicate,
                "edge_color": get_edge_color(op.predicate),
            }
        )
        cyto_edges.append(cyto_edge)
    return cyto_nodes + cyto_edges


STYLE_SHEET = [
    {
        'selector': 'edge',
        'style': {
            'opacity': 1,
            'curve-style': 'unbundled-bezier',
            'taxi-direction': 'vertical',
            'label': 'data(predicate)',
            'target-arrow-shape': 'triangle'
        }
    },

    {
        'selector': 'edge[edge_color]',
        'style': {
            'color': 'data(edge_color)',
            'line-color': 'data(edge_color)',
            'target-arrow-color': 'data(edge_color)'
        }
    },

    {
        'selector': 'node',
        'style': {
            'content': 'data(label)',
            'border-width': 2,
            'text-valign': 'center',
            'padding': "10px",
            'width': 'label',
            'height': '18px',
            'font-size': '18px'
        }
    },

    {
        'selector': "." + ChemicalIdentifier.__name__,
        'style': {
            'shape': 'rectangle',
            'background-color': 'white',
            'text-background-color': 'blue',
        }
    },
    {
        'selector': "." + Action.__name__,
        'style': {
            'shape': 'rectangle',
            # 'background-color': 'none',
            'background-color': 'white',
            'color': 'red',
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

app = Dash(
    name=__name__,
    title="Plan Visualizer",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

cyto.load_extra_layouts()
CYTO = cyto.Cytoscape(
    id='CYTO',
    layout={
        'name': 'cose',
        'nodeDimensionsIncludeLabels': True,
        'animate': True,
        'animationDuration': 1000,
        'align': 'UL',
    },
    style={
        'width': '100%',
        'height': 'calc(100vh - 100px)'
    },
    elements=world_to_cyto(),
    stylesheet=STYLE_SHEET,
)


@app.callback(
    Output("INFO TAB", "children"),
    Input("CYTO", "selectedNodeData"),
    Input("CYTO", "selectedEdgeData"),
)
def update_div_info(node_data, edge_data):
    node_attrs = []
    edge_attrs = []
    if node_data:
        for d in node_data:
            node_attrs.append(d['ele_attrs'])
    if edge_data:
        for d in edge_data:
            u = d['source']
            v = d['target']
            attrs = {
                "source": u,
                "target": v,
                "predicate": d['predicate'],
            }
            edge_attrs.append(attrs)
    blocks = []
    for attrs in node_attrs + edge_attrs:
        try:
            header = attrs['individual class']
        except KeyError:
            header = attrs['predicate']
        json_block = dash_renderjson.DashRenderjson(data=attrs, max_depth=-1,
                                                    theme=JsonTheme, invert_theme=True),
        block = dbc.Card(
            [
                dbc.CardHeader([html.B(header, className="text-primary"), ]),
                dbc.CardBody(json_block),
            ],
            className="mb-3"
        )
        blocks.append(block)
    return blocks


app.layout = html.Div(
    [
        dbc.Row(
            [
                html.Div(
                    [
                        CYTO,
                    ],
                    className="col-lg-8",
                    # style={'height': 'calc(100vh - 100px)'},  # minus header bar height
                    style={'height': '100%'},  # minus header bar height
                ),
                html.Div(
                    [
                        html.H5("Elements", className="text-center mt-3"),
                        html.Hr(),
                        html.Div(id="INFO TAB", className="mt-3"),
                    ],
                    className="col-lg-4",
                ),
            ]
        )

    ], style={"width": "100vw"}
)

if __name__ == '__main__':
    app.run(debug=True)
