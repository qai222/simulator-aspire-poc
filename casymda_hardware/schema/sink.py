import pickle

from casymda.blocks.block_components.block import Block
from simpy import Environment

from hardware_pydantic import Lab
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

        self.sink_counter = 0
        self.sink_log = []

    def do_on_enter(self, job: InstructionJob, previous, current):
        # TODO how does it exactly deal with concurrency?
        # TODO use dedicate logger
        job.notify_job_completion()
        self.time_of_last_entry = self.env.now

        sink_log = "FINISHED\n" + job.name + f" at: {self.time_of_last_entry}\n"
        sink_log += f"{job.instruction.description}\n"
        # sink_log += f"{self.lab['Z1 ARM'].position_on_top_of}\n"
        # sink_log += "-" * 6 + "\n"
        # sink_log += f"{pprint.pformat(self.lab['Z2 ARM'].dict())}"
        # sink_log += f"CURRENT LAB:\n{self.lab}\n"
        # sink_log += "=" * 12
        print(sink_log)
        # TODO resolve pydantic warnings...
        # with open(f"/home/qai/workplace/simulator-aspire-poc/sim_junior/lab_json/state_{self.sink_counter}.json", "w") as f:
        #     f.write(json.dumps(self.lab.model_dump(), indent=2))
        # TODO pydantic is ignoring subclasses when reconstructing (not like in monty the class meta info is kept), have to use pkl fn...
        self.sink_log.append(sink_log)
        self.sink_counter += 1
        with open(f"/home/qai/workplace/simulator-aspire-poc/sim_junior/lab_states/state_{self.sink_counter}.pkl",
                  "wb") as f:
            output = {"simulation time": self.env.now, "lab": self.lab, "log": self.sink_log}
            pickle.dump(output, f)

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
