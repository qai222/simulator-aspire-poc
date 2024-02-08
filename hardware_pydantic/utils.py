from __future__ import annotations

from uuid import uuid4

from loguru import logger


"""Utility functions for hardware_pydantic."""


def str_uuid() -> str:
    """Generate a UUID string.

    Returns
    -------
    str
        A UUID string.
    """
    return str(uuid4())


def resolve_function(func):
    """Resolve a function from a staticmethod.

    Parameters
    ----------
    func : Any
        A function or a staticmethod.
    """
    if isinstance(func, staticmethod):
        return func.__func__
    return func


def action_method_logging(func):
    """The logging decorator for `action_method` of a `Device`.

    Parameters
    ----------
    func : Any
        The function to decorate.

    Returns
    -------
    function_caller : function

     """
    # TODO modify it to fit pre__ and post__

    def function_caller(self, *args, **kwargs):
        _func = resolve_function(func)
        logger.warning(f">> ACTION COMMITTED *{func.__name__}* of *{self.__class__.__name__}*: "
                       f"{self.identifier}")
        for k, v in kwargs.items():
            logger.info(f"action parameter name: {k}")
            logger.info(f"action parameter value: {v}")
        _func(self, *args, **kwargs)

    return function_caller
