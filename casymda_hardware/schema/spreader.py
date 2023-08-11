from casymda.blocks.block_components.block import Block
from simpy.core import Environment

from .device_block import DeviceBlock
from .instruction_job import InstructionJob


class Spreader(Block):
    def __init__(self, env: Environment, device_blocks: list[DeviceBlock]):
        """The conceptual block used for sending jobs to actual devices.

        Parameters
        ----------
        env : Environment
            The `simpy` environment.
        device_blocks : list[DeviceBlock]
            The list of `DeviceBlock`s to which jobs can be sent.

        """
        super().__init__(env, "SPREADER", block_capacity=float('inf'))
        self.device_blocks = device_blocks

    def actual_processing(self, entity: InstructionJob):
        """Process the job by sending it to the next device block.

        Parameters
        ----------
        entity : InstructionJob
            The job to be processed.

        Yields
        ------
        simpy.events.Timeout
            The processing time of the job.

        """
        yield self.env.timeout(0)

    def find_successor(self, job: InstructionJob) -> DeviceBlock:
        """Find the successor of the job.

        Parameters
        ----------
        job : InstructionJob
            The job to be processed.

        Returns
        -------
        DeviceBlock
            The successor of the job.

        Raises
        ------
        ValueError
            If the successor is not found.

        Notes
        -----
        Please note this restricts that a job can only be sent to a `DeviceBlock`.

        """
        for db in self.device_blocks:
            if db.device.identifier == job.get_next_machine():
                return db
        raise ValueError(f'successor not found: {job.get_next_machine()}\nFor: {job.identifier}')
