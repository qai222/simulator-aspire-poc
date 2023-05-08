from casymda.blocks import Sink

from .job import BlockJob


class BlockSink(Sink):
    def __init__(self, env, name, xy=None, ways=None):
        super().__init__(env, name, xy, ways)
        self.do_on_enter_list.append(self.do_on_enter)
        self.time_of_last_entry = -1

    def do_on_enter(self, job: BlockJob, previous, current):
        print("job finished: " + job.name)
        job.notify_job_completion()
        self.time_of_last_entry = self.env.now
