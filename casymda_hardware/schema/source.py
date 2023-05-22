from casymda.blocks.block_components.block import Block
from casymda.blocks.entity import Entity
from simpy.core import Environment
from simpy.events import AllOf, ProcessGenerator

from hardware_pydantic.base import Lab
from .instruction_job import InstructionJob


class Source(Block):
    """
    conceptual block used for creating all jobs
    """

    def __init__(self, env: Environment, lab: Lab):
        super().__init__(env, name="SOURCE", block_capacity=float('inf'))
        self.lab = lab

        env.process(self.creation_loop(self.lab))

    def creation_loop(self, lab: Lab) -> ProcessGenerator:
        # create all jobs
        dict_instruction_job = dict()
        for k, v in lab.dict_instruction.items():
            ins = InstructionJob(env=self.env, lab=lab, instruction=v)
            dict_instruction_job[k] = ins
        dict_instruction_job: dict[str, InstructionJob]

        # let each job subscribe to the completion of the jobs it depends on
        for instruction_job in dict_instruction_job.values():

            preceding_jobs_completion_events = []
            for predecessor_identifier in instruction_job.instruction.preceding_instructions:
                preceding_instruction_job = dict_instruction_job[predecessor_identifier]
                preceding_jobs_completion_events.append(
                    preceding_instruction_job.is_completed_event
                )

            if len(preceding_jobs_completion_events) == 0:
                instruction_job.on_is_ready(None)
            else:
                env = self.env
                all_preceding_jobs_completed = AllOf(
                    env, preceding_jobs_completion_events
                )
                all_preceding_jobs_completed.callbacks.append(
                    instruction_job.is_ready_event.succeed
                )

        # start regular processing
        for instruction_job in dict_instruction_job.values():
            instruction_job.block_resource_request = self.block_resource.request()
            yield instruction_job.block_resource_request
            self.env.process(self.process_entity(instruction_job))

    def actual_processing(self, entity: Entity):
        yield self.env.timeout(0)
