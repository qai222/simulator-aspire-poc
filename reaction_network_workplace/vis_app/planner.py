import dash_bootstrap_components as dbc
import flask
from dash import Dash, html, page_registry, page_container


# import os
# os.environ['LOGURU_LEVEL'] = 'WARNING'


def get_navbar():
    nav_links = []
    for page in page_registry.values():
        description = page['description']
        href = page['relative_path']
        nav_link = dbc.NavLink(
            description, href=href,  # className="mx-2 text-dark", #active="exact",  style={"color": "#ffffff"}
        )
        nav_links.append(nav_link)

    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(
                nl,
            ) for nl in nav_links
        ],
        brand="ASPIRE EPV",
        brand_href="/",
        color="#6f42c1",
        dark=True,
        sticky="top",
        fluid=True,
        className="px-4",
        style={"zIndex": "1"}
    )
    return navbar


def create_dashapp(prefix="/"):
    server = flask.Flask(__name__)

    app = Dash(
        name=__name__,
        title="ASPIRE EPV - case1",
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        use_pages=True,
        server=server,
        # suppress_callback_exceptions=True,
        # assets_ignore=r'defer[A-z]*.js',
        url_base_pathname=prefix,
    )

    # app_folder = os.path.dirname(os.path.abspath(__file__))
    # app._favicon = os.path.join(app_folder, "assets/favicon.ico")

    CONTENT_STYLE = {
        "marginLeft": "2rem",
        "marginRight": "2rem",
        "padding": "1rem 1rem",
        "zIndex": "0",
    }

    content = html.Div(id="page-content", children=[page_container], style=CONTENT_STYLE)

    app.layout = html.Div(
        [
            get_navbar(),
            content,
        ],
        # trick from https://stackoverflow.com/questions/35513264
        style={"width": "100vw"}
    )
    return app


if __name__ == '__main__':
    APP = create_dashapp()
    # APP.run(host="0.0.0.0", port=8070, debug=True)
    APP.run(host="0.0.0.0", port=8070, debug=False)
