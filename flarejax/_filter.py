import dataclasses
import functools
from typing import Any, Callable, ParamSpec, TypeVar

import jax

from ._module import Module

__all__ = [
    "filter_jit",
]

T = TypeVar("T")
P = ParamSpec("P")


@dataclasses.dataclass(frozen=True)
class Arguments(Module):
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Result(Module):
    value: Any


@functools.partial(jax.jit, static_argnames=("function",))
def _apply_filter_jit(module: Arguments, function: Any) -> Any:
    result = function(*module.args, **module.kwargs)
    return Result(result)


def filter_jit(function: Callable[P, T]) -> Callable[P, T]:
    """
    Apply a `jax.jit` to the function but figure out the static arguments
    only treat all leaves exept `jax.Array`s as static arguments.

    Parameters
    ---
    function: Callable
        Function to apply the `jax.jit` to.

    Returns
    ---
    compiled: Callable
        The compiled function. This function takes the same arguments as the
        original function.
    """

    @functools.wraps(function)
    def compiled(*args: P.args, **kwargs: P.kwargs) -> T:
        module = Arguments(args, kwargs)
        return _apply_filter_jit(module, function).value

    return compiled
