from collections import OrderedDict

from casymda.blocks.block_components import VisualizableBlock
from casymda.blocks.entity import Entity
from simpy.core import Environment
from simpy.events import ProcessGenerator

from .job import BlockJob

job_data: dict[str, OrderedDict[str, float]] = {
    "product_1": OrderedDict([("A1", 1), ("A2", 3)]),
    "product_2": OrderedDict([("A1", 1), ("A2", 3)]),
    "product_3": OrderedDict([("A1", 1), ("A2", 3)]),
    "product_4": OrderedDict([("A1", 1), ("A2", 3)]),
    "product_5": OrderedDict([("A1", 1), ("A2", 3)]),
}


class BlockSource(VisualizableBlock):
    def __init__(
            self,
            env: Environment,
            name: str,
            xy: tuple[int, int] = ...,
            ways: dict[str, list[tuple[int, int]]] = ...,
    ):
        super().__init__(env, name, xy=xy, ways=ways)
        env.process(self.creation_loop())

    def creation_loop(self) -> ProcessGenerator:
        # create all jobs
        jobs: dict[str, BlockJob] = {}
        for job_id, machine_times in job_data.items():
            job = BlockJob(
                self.env,
                job_id,
                machines_times=machine_times,
                dependencies=(),
            )
            jobs[job_id] = job

        # start regular processing
        for job in jobs.values():
            job.block_resource_request = self.block_resource.request()
            yield job.block_resource_request
            self.env.process(self.process_entity(job))

    def actual_processing(self, entity: Entity):
        yield self.env.timeout(0)
