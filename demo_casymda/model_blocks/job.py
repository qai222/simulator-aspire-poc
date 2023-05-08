from collections import OrderedDict

from casymda.blocks.entity import Entity
from simpy.core import Environment
from simpy.events import Event, EventCallback


class BlockJob(Entity):
    def __init__(
            self,
            env: Environment,
            name: str,
            machines_times: OrderedDict[str, float],
            dependencies: tuple[str],
    ):
        super().__init__(env, name)
        self._machines_times = machines_times
        self.dependencies = dependencies
        self._num_completed_machines = 0
        self.is_completed_event = Event(env)
        self.is_ready = False
        self.is_ready_event = Event(env)
        self.add_on_is_ready_callback(self.on_is_ready)

    def notify_processing_step_completion(self) -> None:
        self._num_completed_machines += 1

    def notify_job_completion(self) -> None:
        self.is_completed_event.succeed()

    def add_on_is_ready_callback(self, callback: EventCallback):
        self.is_ready_event.callbacks.append(callback)

    def on_is_ready(self, event):
        self.is_ready = True

    def has_next_machine(self) -> bool:
        return len(self._machines_times) > self._num_completed_machines

    def get_next_machine(self) -> str:
        return list(self._machines_times.items())[self._num_completed_machines][0]

    def get_next_processing_time(self) -> float:
        return self._machines_times[self.get_next_machine()]
