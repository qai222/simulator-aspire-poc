from casymda.blocks.block_components import VisualizableBlock

from .job import BlockJob


class BlockMachine(VisualizableBlock):
    def __init__(
            self,
            env,
            name,
            xy=...,
            ways=...,
    ):
        # block_capacity=1, so that every resource can only process 1 job at a time
        super().__init__(env, name, block_capacity=1, xy=xy, ways=ways)

    def actual_processing(self, job: BlockJob):
        assert job.get_next_machine() == self.name
        processing_time = job.get_next_processing_time()
        yield self.env.timeout(processing_time)
        job.notify_processing_step_completion()
