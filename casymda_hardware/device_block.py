from casymda.blocks.block_components.block import Block
from simpy.core import Environment

from casymda_hardware.instruction_job import InstructionJob
from hardware_pydantic import Device


class DeviceBlock(Block):

    def __init__(
            self, env: Environment,
            device: Device,
            block_capacity=1,
    ):
        self.device = device
        self.identifier = self.__class__.__name__ + ": " + self.device.identifier
        super().__init__(env, name=self.identifier, block_capacity=block_capacity)

    def actual_processing(self, job: InstructionJob):
        assert job.get_next_machine() == self.device.identifier
        processing_time = self.device.act_by_instruction(job.instruction)
        # processing_time = entity.get_next_processing_time()
        yield self.env.timeout(processing_time)
        job.notify_processing_step_completion()
