from typing import TypedDict

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


class CytoEdgeData(TypedDict):
    id: str
    source: str
    target: str


class CytoEdge(TypedDict):
    data: CytoEdgeData
    classes: str
    group: str  # edges


class CytoNodeData(TypedDict):
    id: str
    label: str
    url: str


class CytoNode(TypedDict):
    data: CytoNodeData
    classes: str
    group: str  # nodes


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
        'selector': '.transform_reaction',
        'style': {
            'shape': 'rectangle',
            "background-color": "red",
        }
    },
    {
        'selector': '.transform_liquid_addition',
        'style': {
            'shape': 'rectangle',
            "background-color": "gray",
        }
    },
    {
        'selector': '.transform_solid_addition',
        'style': {
            'shape': 'rectangle',
            "background-color": "black",
        }
    },
    {
        'selector': '.transform_purification',
        'style': {
            'shape': 'rectangle',
            "background-color": "blue",
        }
    },
    {
        'selector': '.transform_loading',
        'style': {
            'shape': 'rectangle',
            "background-color": "brown",
        }
    },
    {
        'selector': '.transform_reloading',
        'style': {
            'shape': 'rectangle',
            "background-color": "brown",
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
