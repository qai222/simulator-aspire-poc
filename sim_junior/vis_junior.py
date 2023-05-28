import glob
import os.path
import pickle
import pprint

import dash_bootstrap_components as dbc
import dash_renderjson
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from hardware_pydantic.junior import *


def get_layout_figure(lab: Lab) -> go.Figure:
    fig = go.Figure()
    fig.update_xaxes(
        # range=[0, 500],
        showgrid=False,
        zeroline=False,
        visible=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        # range=[0, 500],
        showgrid=False,
        zeroline=False,
        visible=False,
        fixedrange=True,
    )
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        margin={
            't': 10, 'b': 10, 'l': 10, 'r': 10
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    z1_arm = lab['Z1 ARM']
    z1_arm: JuniorArmZ1

    z2_arm = lab['Z2 ARM']
    z2_arm: JuniorArmZ2
    assert z1_arm.position_on_top_of != z2_arm.position_on_top_of

    for k, v in lab.dict_object.items():
        if isinstance(v, JuniorSlot):
            x0, y0 = v.layout_position
            x1 = x0 + v.layout_x
            y1 = y0 + v.layout_y

            if v.content is not None:
                fillcolor = "gray"
            else:
                fillcolor = None

            fig.add_shape(
                type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(width=2, ),
                fillcolor=fillcolor,
                name=v.identifier,
            )
            fig.add_trace(
                go.Scatter(
                    x=[x0, x0, x0 + v.layout_x, x0 + v.layout_x],
                    y=[y0, y0 + v.layout_y, y0 + v.layout_y, y0],
                    fill="toself",
                    mode='lines',
                    name='',
                    # hovertemplate='<br>',
                    text=pprint.pformat(v.state, indent=2).replace("\n", "<br>"),
                    opacity=0
                )
            )
            if z1_arm.position_on_top_of == v.identifier:
                bgcolor = "red"
            elif z2_arm.position_on_top_of == v.identifier:
                bgcolor = "blue"
            else:
                bgcolor = None

            fig.add_annotation(x=x0 + v.layout_x / 2, y=y0 + v.layout_y / 2,
                               font={"color": "black"},
                               text=v.identifier,
                               bordercolor=bgcolor,
                               borderwidth=3,
                               showarrow=False,
                               yshift=0)
    return fig


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

state_files = glob.glob("/home/qai/workplace/simulator-aspire-poc/sim_junior/lab_states/state_*.pkl")
STATES = []
for sf in sorted(state_files, key=lambda x: int(os.path.basename(x).replace(".pkl", "").replace("state_", ""))):
    with open(sf, "rb") as f:
        STATE = pickle.load(f)
        STATES.append(STATE)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

card_layout = dbc.Card(
    [
        dbc.CardHeader("Junior Layout"),
        dbc.CardBody(
            dcc.Graph(
                style={'height': '700px'},
                config={'displayModeBar': False},
                id="layout-figure"
            )
        )
    ]
)

col_left = html.Div(card_layout, className="col-7 p-2")


def get_tracker_options():
    """ this persists """
    state = STATES[0]
    lab = state["lab"]
    d = []
    for k in lab.dict_object:
        d.append({
            "label": k,
            "value": k
        })
    return d


card_tracker1 = dbc.Col(dbc.Card(
    [
        dbc.CardHeader("Tracker 1"),
        dbc.CardBody(
            [
                dcc.Dropdown(id="tracker-1-select", options=get_tracker_options(), value="Z1 ARM"),
                dash_renderjson.DashRenderjson(id="tracker-1-json", max_depth=-1, theme=JsonTheme, invert_theme=True, )
            ],
        )
    ]
))

list_group = dbc.ListGroup(id='sim-log',
                           )

card_log = dbc.Col(dbc.Card(
    [
        dbc.CardHeader("Simulation log"),
        dbc.CardBody(
            list_group
        )
    ]
))

card_trackers = [card_tracker1, card_log]

col_right = html.Div([
    dbc.Row(
        card_trackers, className="mb-2"
    ),
], className="col-5 p-2")

content = dbc.Row(
    [
        col_left, col_right
    ],
    className="mt-2 mx-2"
)
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.NavbarBrand("NCATS JUNIOR Simulator", className="ms-2"),
                href="https://github.com/qai222/simulator-aspire-poc",
                style={"textDecoration": "none"}, className="g-0"
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        ], className="m-0"
    ),
    color="dark",
    dark=True,
)

state_controller = html.Div(
    [
        html.H4('State Index:', style={'display': 'inline-block', 'margin-right': 20}),
        dcc.Input(
            id="state-number",
            type="number",
            placeholder="state index",
            value=0,
            max=len(STATES) - 1,
            min=0,
        ),
        html.H5(style={'margin-right': 20}, id="sim-time"),
    ], className="mt-3 mx-3"
)

app.layout = html.Div(
    [
        navbar,
        state_controller,
        content,
    ]
)


@app.callback(
    Output("layout-figure", "figure"),
    Output("sim-time", "children"),
    Output("tracker-1-json", "data"),
    Output("sim-log", "children"),
    Input("state-number", "value"),
    Input("tracker-1-select", "value"),

)
def update_layout_figure(i_state: int, tracker_1_id):
    if i_state > len(STATES) - 1:
        i_state = len(STATES) - 1
    loaded_state = STATES[i_state]
    current_time = loaded_state['simulation time']
    current_lab = loaded_state['lab']
    fig_layout = get_layout_figure(current_lab)
    tracker_1_obj = current_lab[tracker_1_id]

    sim_log = loaded_state['log']
    log_items = [
        dbc.ListGroupItem(l) for l in sim_log
    ]

    return fig_layout, "Simulation time: {}".format(current_time), tracker_1_obj.model_dump(), log_items


if __name__ == '__main__':
    app.run_server()
