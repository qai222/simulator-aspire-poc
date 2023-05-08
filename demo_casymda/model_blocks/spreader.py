from casymda.blocks.block_components import VisualizableBlock

from .job import BlockJob
from .util import block_with_name


class BlockSpreader(VisualizableBlock):
    def __init__(
            self,
            env,
            name,
            xy=...,
            ways=...,
    ):
        super().__init__(env, name, xy=xy, ways=ways)

    def actual_processing(self, entity: BlockJob):
        yield self.env.timeout(0)

    def find_successor(self, job: BlockJob):
        return block_with_name(self.successors, job.get_next_machine())
