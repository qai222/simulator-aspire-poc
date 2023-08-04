from casymda.blocks.block_components.block import Block

from .buffer import Buffer
from .instruction_job import InstructionJob
from .sink import Sink


class Check(Block):
    def __init__(self, env, sink: Sink, buffer: Buffer):
        """Check block.

        Parameters
        ----------
        env : Environment
            The `simpy` environment for the simulation.

        """
        super().__init__(env, "CHECK", block_capacity=float('inf'))
        self.sink = sink
        self.buffer = buffer

    def actual_processing(self, job: InstructionJob):
        """Processing the jobs with timeout 0.

        Parameters
        ----------
        job : InstructionJob
            The job to process.

        Yields
        ------
        simpy.events.Timeout
            The timeout event of the processing.

        """
        yield self.env.timeout(0)

    def find_successor(self, job: InstructionJob) -> Block:
        """Find the successor of the job.

        Parameters
        ----------
        job : InstructionJob
            The job to find the successor for.

        Returns
        -------
        Block
            Returns the sink if the job is done, otherwise the buffer.

        """
        job_is_done = not job.has_next_machine()
        if job_is_done:
            return self.sink
        else:
            return self.buffer
