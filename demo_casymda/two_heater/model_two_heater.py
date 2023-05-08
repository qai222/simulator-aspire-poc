from model_blocks import BlockSource, BlockSink, BlockMachine, BlockSpreader

class Model:
    """generated model"""

    def __init__(self, env):
        self.env = env

        #!resources+components

        self.source = BlockSource(
            self.env, "source", xy=(24, 113), ways={"spreader": [(42, 113), (269, 113)]}
        )

        self.sink = BlockSink(self.env, "sink", xy=(884, 113), ways={})

        self.A1 = BlockMachine(
            self.env,
            "A1",
            xy=(474, 43),
            ways={"sink": [(524, 43), (695, 43), (695, 113), (866, 113)]},
        )

        self.A2 = BlockMachine(
            self.env,
            "A2",
            xy=(474, 193),
            ways={"sink": [(524, 193), (695, 193), (695, 113), (866, 113)]},
        )

        self.spreader = BlockSpreader(
            self.env,
            "spreader",
            xy=(294, 113),
            ways={
                "A1": [(294, 88), (294, 43), (424, 43)],
                "A2": [(294, 138), (294, 193), (424, 193)],
            },
        )

        #!model

        self.model_components = {
            "source": self.source,
            "sink": self.sink,
            "A1": self.A1,
            "A2": self.A2,
            "spreader": self.spreader,
        }

        self.model_graph_names = {
            "source": ["spreader"],
            "sink": [],
            "A1": ["sink"],
            "A2": ["sink"],
            "spreader": ["A1", "A2"],
        }
        # translate model_graph_names into corresponding objects
        self.model_graph = {
            self.model_components[name]: [
                self.model_components[nameSucc]
                for nameSucc in self.model_graph_names[name]
            ]
            for name in self.model_graph_names
        }

        for component in self.model_graph:
            component.successors = self.model_graph[component]
