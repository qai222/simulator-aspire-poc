from casymda.blocks.entity import Entity
from simpy.core import Environment
from simpy.events import Event, EventCallback

from hardware_pydantic import Lab, Instruction


class InstructionJob(Entity):
    """
    bijective to a `hardware.Instruction` instance
    
    technically this is `casymda_jobshop.Job` but there is at least one important difference:
    `Job` describes either 
    - one `Instruction` instance, or 
    - a set of *linearly* connected `Instruction` instances (like a path graph)
    so far I do not find the latter necessary or greatly helpful, so this class implements only the former definition.
    
    this means
    - `_num_completed_machines` can only be 0 (init) or 1 (finished)
    - there would be at most one "next_machine"
        - note about "next_machine": in `casymda_jobshop.Job` the concept "next_machine" means the `Block` instance
            defined for **this `Job` instance** that it has not gone through. it never points to a `Block` instance
            defined for another `Job` instance
    #TODO serialization
    #TODO visualization
    """

    def __init__(self, env: Environment, lab: Lab, instruction: Instruction):
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
        return [self.lab.dict_instruction[ii] for ii in self.instruction.preceding_instructions]

    def notify_processing_step_completion(self) -> None:
        self._num_completed_machines += 1

    def notify_job_completion(self) -> None:
        self.is_completed_event.succeed()

    def add_on_is_ready_callback(self, callback: EventCallback):
        self.is_ready_event.callbacks.append(callback)

    def on_is_ready(self, event: Event):
        self.is_ready = True
        # TODO dedicate logger
        print(self.instruction.identifier, f" ready at: {self.env.now}")

    def has_next_machine(self) -> bool:
        # return len(self._machines_times) > self._num_completed_machines
        return self._num_completed_machines < 1

    def get_next_machine(self) -> str:
        # return list(self._machines_times.items())[self._num_completed_machines][0]
        return self.instruction.actor_device.identifier
