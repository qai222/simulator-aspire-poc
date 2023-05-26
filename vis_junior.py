import pprint

import plotly.graph_objects as go
from dash import Dash, dcc, html

from hardware_pydantic.junior import *


def layout_add_slots(fig):
    for k, v in JUNIOR_LAB.dict_object.items():
        if isinstance(v, JuniorSlot):
            x0, y0 = v.layout_position
            x1 = x0 + v.layout_x
            y1 = y0 + v.layout_y
            fig.add_shape(
                type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(width=0, ),
                fillcolor="LightSkyBlue",
                name=v.identifier
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
            fig.add_annotation(x=x0 + v.layout_x / 2, y=y0 + v.layout_y / 2,
                               text=v.identifier,
                               showarrow=False,
                               yshift=0)
    fig.update_xaxes(
        # range=[0, 500],
        showgrid=False,
        zeroline=False,
        visible=False,
    )
    fig.update_yaxes(
        # range=[0, 500],
        showgrid=False,
        zeroline=False,
        visible=False,
    )
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )


if __name__ == '__main__':
    create_junior_base()
    app = Dash(__name__)
    fig = go.Figure()
    layout_add_slots(fig)
    graph = dcc.Graph(
        figure=fig,
        style={'width': '90vw', 'height': '90vh'}
    )
    app.layout = html.Div([graph])
    app.run_server()
