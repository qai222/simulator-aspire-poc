from casymda.blocks.block_components.block import Block
from simpy import Interrupt
from simpy.core import Environment

from .instruction_job import InstructionJob


class Buffer(Block):
    def __init__(self, env: Environment):
        """
        conceptual block used for sending jobs to actual devices
        # TODO combine with `Spreader`
        """
        super().__init__(env, name="BUFFER", block_capacity=float('inf'))

    def actual_processing(self, job: InstructionJob):
        """
        (flad's original comments:)
        just release every job as soon as it is ready
        tbd: alternatively let a dedicated scheduler release specific jobs at certain times
        """
        if job.is_ready:
            yield self.env.timeout(0)
        else:
            # wake up on is_ready
            job.add_on_is_ready_callback(lambda ev: job.current_process.interrupt())
            # wait
            try:
                yield self.env.timeout(float("inf"))
            except Interrupt:
                pass  # job is_ready callback executed
