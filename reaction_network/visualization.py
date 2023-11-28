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
            'width': 800,
            'height': 300,
            'shape': 'rectangle',
            'background-fit': 'contain',
            'background-image': 'data(url)',
            "border-width": "6px",
            "border-color": "black",
            "border-opacity": "1.0",
            "background-color": "white",
        }
    },
    {
        'selector': '.compound',
        'style': {
            'width': 180,
            'height': 100,
            'shape': 'circle',
            'background-fit': 'contain',
            'background-image': 'data(url)',
            "border-width": "6px",
            "border-color": "black",
            "border-opacity": "1.0",
            "background-color": "white",
            # "content": 'data(label)',
            # "text-outline-color": "#77828C"
        }
    },
    {
        'selector': '.compound_starting',
        'style': {
            "background-color": "#cff0fa",
        }
    },
    {
        'selector': '.compound_intermediate',
        'style': {
            "background-color": "#f0facf",
        }
    },
    {
        'selector': '.compound_target',
        'style': {
            "background-color": "#ffaba3",
        }
    },
    {
        'selector': '.transform_reaction',
        'style': {
            'shape': 'rectangle',
            "background-color": "#FF0000",
        }
    },
    {
        'selector': '.transform_liquid_addition',
        'style': {
            'shape': 'rectangle',
            "background-color": "#808080",
        }
    },
    {
        'selector': '.transform_solid_addition',
        'style': {
            'shape': 'rectangle',
            "background-color": "#000000",
        }
    },
    {
        'selector': '.transform_purification',
        'style': {
            'shape': 'rectangle',
            "background-color": "#0000FF",
        }
    },
    {
        'selector': '.transform_loading',
        'style': {
            'shape': 'rectangle',
            "background-color": "#800020",
        }
    },
    {
        'selector': '.transform_reloading',
        'style': {
            'shape': 'rectangle',
            "background-color": "#800020",
        }
    },
    {
        'selector': ':selected',
        'style': {
            'z-index': 1000,
            # 'background-color': 'SteelBlue',
            'border-opacity': "1.0",
            "border-color": "SteelBlue",
            'line-color': 'SteelBlue',
            "border-width": "8px",
        }
    },
]
