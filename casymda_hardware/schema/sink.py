from casymda.blocks.block_components.block import Block
from hardware_pydantic import Lab
from simpy import Environment

from .instruction_job import InstructionJob


class Sink(Block):
    def __init__(self, env: Environment, lab: Lab):
        """
        conceptual block used for sending jobs to actual devices
        """
        super().__init__(env, name="SINK", block_capacity=float('inf'))
        self.do_on_enter_list.append(self.do_on_enter)
        self.time_of_last_entry = -1
        self.lab = lab

    def do_on_enter(self, job: InstructionJob, previous, current):
        # TODO use dedicate logger
        job.notify_job_completion()
        self.time_of_last_entry = self.env.now

        sink_log = "FINISHED STEP\n" + job.name + f" at: {self.time_of_last_entry}"
        sink_log += "-" * 6
        sink_log += f"CURRENT LAB:\n{self.lab}"
        sink_log += "=" * 12
        print(sink_log)

    def process_entity(self, entity):
        yield self.env.timeout(0)

        entity.time_of_last_arrival = self.env.now
        self.on_enter(entity)
        self.overall_count_in += 1
        self.entities.append(entity)
        self.block_resource.release(entity.block_resource_request)
        self.on_exit(entity, None)

    def actual_processing(self, entity):
        """not called in this special block"""
