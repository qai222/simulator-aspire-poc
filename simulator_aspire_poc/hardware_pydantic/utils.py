from __future__ import annotations

from uuid import uuid4

from loguru import logger


def str_uuid() -> str:
    return str(uuid4())


def resolve_function(func):
    if isinstance(func, staticmethod):
        return func.__func__
    return func


def action_method_logging(func):
    # TODO modify it to fit pre__ and post__
    """ logging decorator for `action_method` of a `Device` """

    def function_caller(self, *args, **kwargs):
        _func = resolve_function(func)
        logger.warning(f">> ACTION COMMITTED *{func.__name__}* of *{self.__class__.__name__}*: {self.identifier}")
        for k, v in kwargs.items():
            logger.info(f"action parameter name: {k}")
            logger.info(f"action parameter value: {v}")
        _func(self, *args, **kwargs)

    return function_caller
