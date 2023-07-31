import pickle
import pprint
import math

import dash_bootstrap_components as dbc
import dash_renderjson
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import plotly.figure_factory as ff
from hardware_pydantic.junior import *
import plotly.express as px
import pandas as pd


def get_gantt_fig(states):
    df_data = []

    for state in states:
        if state['instruction'] is None:
            continue
        start = state['last_entry']
        end = state['finished']
        task = state['instruction'].identifier
        des = state['instruction'].description

        gantt_task = des.split(":")[0]
        if gantt_task.startswith("wait"):
            gantt_task = "wait"

        df_data.append(
            {"Task": gantt_task, "Start": start, "des": des, "Finish": end, "cost": end - start}
        )
    if len(df_data) == 0:
        return go.Figure()
    df = pd.DataFrame(df_data)

    colors = px.colors.qualitative.G10
    keys = sorted(df['Task'].unique())
    assert len(keys) <= len(colors)
    colors = dict(zip(keys, colors[:len(keys)]))

    fig = ff.create_gantt(df, index_col='Task', bar_width=0.4, show_colorbar=True, colors=colors)
    fig.update_layout(xaxis_type='linear', autosize=True, yaxis_visible=False)
    return fig

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

    arm_platform = lab['ARM PLATFORM']
    arm_platform: JuniorArmPlatform

    for k, v in lab.dict_object.items():
        if isinstance(v, (JuniorSlot, JuniorWashBay, JuniorTipDisposal)):
            x0, y0 = v.layout.layout_position
            x1 = x0 + v.layout.layout_x
            y1 = y0 + v.layout.layout_y

            if isinstance(v, JuniorSlot) and v.slot_content['SLOT'] is not None:
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
                    x=[x0, x0, x0 + v.layout.layout_x, x0 + v.layout.layout_x],
                    y=[y0, y0 + v.layout.layout_y, y0 + v.layout.layout_y, y0],
                    fill="toself",
                    mode='lines',
                    name='',
                    # hovertemplate='<br>',
                    text=pprint.pformat(v.state, indent=2).replace("\n", "<br>"),
                    opacity=0
                )
            )
            if arm_platform.anchor_arm == z1_arm.identifier and arm_platform.position_on_top_of == v.identifier:
                bgcolor = "red"
            elif arm_platform.anchor_arm == z2_arm.identifier and arm_platform.position_on_top_of == v.identifier:
                bgcolor = "blue"
            else:
                bgcolor = None

            fig.add_annotation(x=x0 + v.layout.layout_x / 2, y=y0 + v.layout.layout_y / 2,
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

with open("sim_con-4.pkl", "rb") as f:
    sim_logs = pickle.load(f)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="JUNIOR SIMULATOR")

card_layout = dbc.Card(
    [
        dbc.CardHeader("Junior Layout"),
        dbc.CardBody(
            dcc.Graph(
                style={'height': '600px'},
                config={'displayModeBar': False},
                id="layout-figure",
            ),
            style={'height': '600px'},
        )
    ]
)

col_left = html.Div(
    [
        card_layout,
        dcc.Graph(
            style={'height': '700px'},
            config={'displayModeBar': False},
            id="gantt-figure"
        )
    ],
    className="col-7 p-2"
)


def get_tracker_options(states):
    """ this persists """
    state = states[0]
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
        dbc.CardHeader("Object Tracker"),
        dbc.CardBody(
            [
                dcc.Dropdown(id="tracker-1-select", options=get_tracker_options(sim_logs), value="Z2 ARM"),
                dash_renderjson.DashRenderjson(id="tracker-1-json", max_depth=-1, theme=JsonTheme, invert_theme=True, )
            ],
        )
    ], className="h-100"
))

list_group = dbc.ListGroup(id='sim-log', )

card_log = dbc.Col(dbc.Card(
    [
        dbc.CardHeader("Simulation log"),
        dbc.CardBody(
            list_group, style={"overflow": "scroll", "height": 600}
        )
    ],
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
            max=len(sim_logs) - 1,
            min=0,
        ),
        html.H5(style={'margin-right': 20}, id="sim-time"),
        # dcc.Slider(0, sim_logs[-1]['finished'], value=0, marks={j['finished']: {'label': ""} for j in sim_logs[1:]}, id='sim-slider'),
        # TODO add this
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
    Output("gantt-figure", "figure"),
    Input("state-number", "value"),
    Input("tracker-1-select", "value"),
)
def update_layout_figure(i_state: int, tracker_1_id):
    if i_state > len(sim_logs) - 1:
        i_state = len(sim_logs) - 1
    loaded_state = sim_logs[i_state]
    current_time = loaded_state['finished']
    current_lab = loaded_state['lab']
    fig_layout = get_layout_figure(current_lab)
    tracker_1_obj = current_lab[tracker_1_id]

    log_items = []

    for state in sim_logs[:i_state + 1]:
        ins = state['instruction']
        if ins is None:
            ins_id = None
            ins_des = None
        else:
            ins_id = ins.identifier
            ins_des = ins.description
        item_content = [
            html.B("SIM TIME "),
            f"{state['finished']}",
            html.Br(),
            html.B("FINISHED INSTRUCTION "),
            html.Br(),
            ins_id,
            html.Br(),
            html.B("DESCRIPTION "),
            html.Br(),
            ins_des
        ]
        log_items.append(
            dbc.ListGroupItem(item_content)
        )
    fig_gantt = get_gantt_fig(sim_logs[:i_state+1])
    return fig_layout, "Simulation time: {}".format(current_time), tracker_1_obj.model_dump(), log_items, fig_gantt


if __name__ == '__main__':
    app.run_server(port=8049)
