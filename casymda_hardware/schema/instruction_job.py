from casymda.blocks.entity import Entity
from simpy.core import Environment
from simpy.events import Event, EventCallback

from hardware_pydantic import Lab, Instruction


class InstructionJob(Entity):
    def __init__(self, env: Environment, lab: Lab, instruction: Instruction):
        """The instruction job which is bijective to a `hardware.Instruction` instance.

        Parameters
        ----------
        env : Environment
            The `simpy` environment.
        lab : Lab
            The `hardware_pydantic.Lab` instance.
        instruction : Instruction
            The `hardware_pydantic.Instruction` instance.

        Notes
        -----
        Technically speaking,  this is `casymda_jobshop.Job` but there is at least one important
        difference: `Job` describes either one `Instruction` instance, or a set of
        *linearly* connected `Instruction` instances (like a path graph).
        We only implement the former case here.

        this means
        - `_num_completed_machines` can only be 0 (init) or 1 (finished)
        - there would be at most one "next_machine"
            - note about "next_machine": in `casymda_jobshop.Job` the concept "next_machine" means
            the `Block` instance defined for **this `Job` instance** that it has not gone
            through. it never points to a `Block` instance defined for another `Job` instance.

        """
        #TODO serialization
        #TODO visualization

        self.instruction = instruction
        self.identifier = self.__class__.__name__ + ": " + self.instruction.identifier
        self.lab = lab

        super().__init__(env, name=self.identifier)

        # copy `Job` from `casymda-job-shop-with-precedence`
        self._num_completed_machines = 0
        self.is_completed_event = Event(env)
        self.is_ready = False
        self.is_ready_event = Event(env)
        self.add_on_is_ready_callback(self.on_is_ready)

    @property
    def preceding_instructions(self) -> list[Instruction]:
        """The preceding instructions of this instruction job.

        Returns
        -------
        list[Instruction]
            The preceding instructions of this instruction job.

        """
        return [self.lab.dict_instruction[ii] for ii in self.instruction.preceding_instructions]

    def notify_processing_step_completion(self) -> None:
        """Notify that a processing step has been completed."""
        self._num_completed_machines += 1

    def notify_job_completion(self) -> None:
        """Notify that the job has been completed."""
        self.is_completed_event.succeed()

    def add_on_is_ready_callback(self, callback: EventCallback):
        """Add a callback to the `is_ready_event`."""
        self.is_ready_event.callbacks.append(callback)

    def on_is_ready(self, event: Event):
        """Set the `"""
        self.is_ready = True
        # TODO dedicate logger
        # print(self.instruction.identifier, f" ready at: {self.env.now}")

    def has_next_machine(self) -> bool:
        """Whether there is a next machine to process this job."""
        return self._num_completed_machines < 1

    def get_next_machine(self) -> str:
        """Get the next machine to process this job."""
        return self.instruction.device.identifier
