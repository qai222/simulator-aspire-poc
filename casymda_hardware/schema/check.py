from casymda.blocks.block_components.block import Block

from .buffer import Buffer
from .instruction_job import InstructionJob
from .sink import Sink


class Check(Block):
    def __init__(self, env, sink: Sink, buffer: Buffer):
        super().__init__(env, "CHECK", block_capacity=float('inf'))
        self.sink = sink
        self.buffer = buffer

    def actual_processing(self, job: InstructionJob):
        yield self.env.timeout(0)

    def find_successor(self, job: InstructionJob) -> Block:
        job_is_done = not job.has_next_machine()
        if job_is_done:
            return self.sink
        else:
            return self.buffer
