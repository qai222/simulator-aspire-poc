from casymda.blocks.block_components.block import Block
from simpy.core import Environment

from hardware_pydantic import Device
from .instruction_job import InstructionJob
from .object_resource import LabObjectResource


class DeviceBlock(Block):

    def __init__(
            self,
            env: Environment,
            device: Device,
            block_capacity=1,
    ):
        """Device block, which can be used to model a device in a lab.

        Parameters
        ----------
        env : Environment
            The `simpy` environment.
        device : Device
            The device to be modeled.
        block_capacity : int, optional
            The capacity of the block. Default is 1.

        """
        self.device = device
        self.identifier = self.__class__.__name__ + ": " + self.device.identifier
        super().__init__(env, name=self.identifier, block_capacity=block_capacity)

    def actual_processing(self, job: InstructionJob):
        """The actual processing of the job.

        Parameters
        ----------
        job : InstructionJob
            The job to be processed.

        Notes
        -----
        This process involves the following steps:
        1. Check if we have the right device as resource;
        2. Make projections;
        3. Request resources for lab objects;
        4. Run preactor check to make sure everything is ready;
        5. Move clock;
        6. Release resources;
        7. Exit and change job status.

        """
        # check we are using the right device
        assert job.get_next_machine() == self.device.identifier

        # make projections
        involved_objects, processing_time = self.device.act_by_instruction(job.instruction, actor_type="proj")

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
