from casymda.blocks.block_components.block import Block
from simpy.core import Environment

from hardware_pydantic import Device
from .instruction_job import InstructionJob
from .object_resource import LabObjectResource


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
        involved_objects, processing_time = self.device.act_by_instruction(job.instruction, is_pre=True)
        print(f"projected processing time: {job.identifier} == {processing_time}")
        print(f"involved objects: {involved_objects}")
        # TODO test object resources are working
        resource_objects = [LabObjectResource.from_lab_object(obj, self.env) for obj in involved_objects]
        reqs = [ro.resource.request() for ro in resource_objects]
        for req in reqs:
            yield req
        yield self.env.timeout(processing_time)
        for i, ro in enumerate(resource_objects):
            req = reqs[i]
            ro.resource.release(req)
        self.device.act_by_instruction(job.instruction, is_pre=False)
        job.notify_processing_step_completion()
