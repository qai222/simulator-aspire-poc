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
        # check we are using the right device
        assert job.get_next_machine() == self.device.identifier

        # make projections
        involved_objects, processing_time = self.device.act_by_instruction(job.instruction, actor_type="proj")
        # TODO should projection be made before sending the request or after requests have been accepted (or both)

        # request resources for lab objects
        resource_objects = [LabObjectResource.from_lab_object(obj, self.env) for obj in involved_objects]
        reqs = [ro.resource.request() for ro in resource_objects]
        # note the device resource is requested/released in `_process_entity` of `Block`
        for req in reqs:
            yield req

        # everything is ready, run preactor check
        self.device.act_by_instruction(job.instruction, actor_type="pre")
        # move clock
        yield self.env.timeout(processing_time)
        self.device.act_by_instruction(job.instruction, actor_type="post")
        # release resources
        for i, ro in enumerate(resource_objects):
            req = reqs[i]
            ro.resource.release(req)
        # exit, change job status
        job.notify_processing_step_completion()
