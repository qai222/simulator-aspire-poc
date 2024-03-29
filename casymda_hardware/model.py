from __future__ import annotations

import os

from simpy import Environment

from hardware_pydantic import *
from .schema import Source, Buffer, Spreader, Check, Sink, DeviceBlock


class Model:
    def __init__(self, env: Environment, lab: Lab, wdir: str | os.PathLike, model_name: str):
        """Model class for the casymda hardware.

        Parameters
        ----------
        env : Environment
            The simpy environment.
        lab : Lab
            The lab object.
        wdir : str | os.PathLike
            The working directory.
        model_name : str
            The name of the model.

        """
        self.env = env
        self.lab = lab

        # !resources+components
        self.source = Source(self.env, self.lab)
        self.sink = Sink(self.env, self.lab, wdir, model_name)
        self.buffer = Buffer(self.env)

        self.device_blocks = []
        for i, device in lab.dict_object.items():
            if isinstance(device, Device):
                device_block = DeviceBlock(self.env, device, block_capacity=1)
                self.device_blocks.append(device_block)

        self.spreader = Spreader(self.env, device_blocks=self.device_blocks)

        self.check = Check(self.env, self.sink, self.buffer)

        # !model
        self.source.successors = [self.buffer, ]
        self.buffer.successors = [self.spreader, ]
        self.spreader.successors = self.device_blocks
        for db in self.device_blocks:
            db.successors = [self.check, ]
        self.check.successors = [self.sink, self.buffer]
