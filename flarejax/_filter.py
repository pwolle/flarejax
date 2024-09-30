"""
Ease the use of jax.jit by automatically figuring out the static arguments
"""

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
    Apply a jax.jit to the function but figure out the static arguments
    only treat all leaves exept jax.Arrays as static arguments.

    With plain jax.jit all PyTree leaves are traced by default, this can be
    a problem if the leaves are not valid jax types (e.g. strings) or if the
    control flow of the function depends on them.

    This could be solved by always making sure that all leaves are valid jax
    types, but this would make it impossible to use the jax.tree_util functions
    to manipulate other parts of the PyTree.

    Since inputs (or its leaves) are converted to jax.Arrays in the
    regular jax.jit, a good middle ground is to treat all leaves except
    jax.Arrays as static arguments. This way the control flow can depend on
    all non-jax.Array leaves.

    This however has the drawback that the function will be recompiled for each
    unique combination non-jax.Array leaves. To avoid this the leaces should
    be cast to jax.Arrays before calling the function.

    A further recommendation is to only use hashable types in the pytree leaves,
    to avoid unnecessary recompilations.

    Parameters
    ---
    function: Callable
        Function to apply the jax.jit to.

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
