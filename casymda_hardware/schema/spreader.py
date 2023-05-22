from casymda.blocks.block_components.block import Block
from simpy.core import Environment

from .device_block import DeviceBlock
from .instruction_job import InstructionJob


class Spreader(Block):
    def __init__(self, env: Environment, device_blocks: list[DeviceBlock]):
        """
        conceptual block used for sending jobs to actual devices
        """
        super().__init__(env, "SPREADER", block_capacity=float('inf'))
        self.device_blocks = device_blocks

    def actual_processing(self, entity: InstructionJob):
        yield self.env.timeout(0)

    def find_successor(self, job: InstructionJob) -> DeviceBlock:
        """ note this restricts that a job can only be sent to a `DeviceBlock` """
        for db in self.device_blocks:
            if db.device.identifier == job.get_next_machine():
                return db
        raise ValueError(f'successor not found: {job.get_next_machine()}\nFor: {job.identifier}')
