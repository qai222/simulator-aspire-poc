from __future__ import annotations

import os
import pickle

from casymda.blocks.block_components.block import Block
from simpy import Environment
from copy import deepcopy
from hardware_pydantic import Lab
from .instruction_job import InstructionJob


class Sink(Block):
    def __init__(self, env: Environment, lab: Lab, wdir: str | os.PathLike, model_name: str):
        """
        conceptual block used for sending jobs to actual devices
        """
        super().__init__(env, name="SINK", block_capacity=float('inf'))
        self.model_name = model_name
        self.do_on_enter_list.append(self.do_on_enter)
        self.do_on_exit_list.append(self.do_on_exit)
        self.time_of_last_last_entry = -1
        self.time_of_last_entry = -1
        self.lab = lab

        self.wdir = wdir
        self.sink_counter = 0
        self.sink_log = [
            {
                "finished": self.time_of_last_entry,
                "last_entry": self.time_of_last_last_entry,
                "state_index": self.sink_counter,
                "instruction": None,
                "lab": deepcopy(self.lab),
            }
        ]

    def do_on_exit(self, job: InstructionJob, previous, current):
        sink_log = {
            "finished": self.time_of_last_entry,
            "last_entry": self.time_of_last_last_entry,
            "state_index": self.sink_counter,
            "instruction": job.instruction,
            "lab": deepcopy(self.lab),
        }
        self.sink_log.append(sink_log)
        print(sink_log['last_entry'], sink_log['finished'], sink_log['instruction'].description)
        # TODO pydantic is ignoring subclasses when reconstructing (not like in monty the class meta info is kept), have to use pkl fn...
        # TODO use dedicate logger
        with open(os.path.join(f"{self.wdir}", f"sim_{self.model_name}.pkl"), "wb") as f:
            # output = {"simulation_time": self.env.now, "lab": self.lab, "instruction": job.instruction, "last_entry": last_entry, "log": self.sink_log}
            pickle.dump(self.sink_log, f)

    def do_on_enter(self, job: InstructionJob, previous, current):
        # TODO how does it exactly deal with concurrency?
        job.notify_job_completion()
        self.time_of_last_last_entry = self.time_of_last_entry
        self.time_of_last_entry = self.env.now

        self.sink_counter += 1

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
