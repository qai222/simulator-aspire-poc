import dash_bootstrap_components as dbc
from dash import html, get_app
from dash import register_page

register_page(__name__, path='/', description="Home")


def get_card(title: str, text: str, link_text: str, link_path: str = "#", width: int = 22):
    card = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H4(title, className="card-title"),
                    html.P(
                        text,
                        className="card-text",
                    ),
                    dbc.Button(link_text, color="primary", href=link_path),
                ]
            ),
        ],
        style={"min-width": f"{width}rem", "width": f"{width}rem"},
        className="mx-4"
    )
    return card


card_operation_graph = get_card(
    title="Operation Graph",
    text="Given a quantified reaction network, visualize the operations that realize the reactions.",
    link_text="Operation Graph",
    link_path="/operation_graph",
    width=20,
)

card_reaction_network = get_card(
    title="Reaction Network",
    text="Given the target compounds and their amounts, visualize the suggested reaction network.",
    link_text="Reaction Network",
    link_path="/reaction_network",
    width=20,
)

card_junior_simulator = get_card(
    title="Workstation Simulator",
    text="Given an operation graph, Simulate its operations on the digital twin of a workstation.",
    link_text="Workstation Simulator",
    link_path="/workstation_simulator",
    width=20,
)

layout = dbc.Row(
    dbc.Col(
        [
            card_reaction_network,
            card_operation_graph,
            card_junior_simulator,
        ],
        className="justify-content-center col-8 d-flex mx-auto"
    ),
    className="align-items-center",
    style={"min-height": "50vh"}
)

app = get_app()
