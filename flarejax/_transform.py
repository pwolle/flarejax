"""
Jax transformations as modules. This allows for the benefits of modules 
(serialization and flattening) to be used with jax transformations.
"""

import dataclasses
from typing import Any, Callable, Generic, Hashable, ParamSpec, TypeVar

import jax

from ._filter import filter_jit
from ._module import Module

__all__ = [
    "Jit",
    "Vmap",
    "Partial",
]

T = TypeVar("T")
P = ParamSpec("P")


# a function which applies a module to its arguments
# this is needed to have the jax.jit compilation cache all in one place
@filter_jit
def _filter_jit_apply(
    module: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    return module(*args, **kwargs)


@dataclasses.dataclass
class Jit(Module, Generic[P, T]):
    """
    Call a module with `filter_jit` applied to the `__call__` method.

    Parameters
    ---
    module: Callable[P, T]
        The module to apply `filter_jit` to.
    """

    module: Callable[P, T]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return _filter_jit_apply(self.module, *args, **kwargs)


@dataclasses.dataclass
class Vmap(Module, Generic[P, T]):
    """
    Call a module with `jax.vmap` applied to the `__call__` method.

    Parameters
    ---
    module: Callable[P, T]
        The module to apply `jax.vmap` to.

    **kwargs: Any
        Arguments to pass to `jax.vmap`.

    """

    module: Callable[P, T]
    in_axes: int | None | tuple[int, ...] = 0
    out_axes: Any = 0
    axis_name: Hashable | None = None
    axis_size: int | None = None
    spmd_axis_name: Hashable | tuple[Hashable, ...] = None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return jax.vmap(
            self.module,
            in_axes=self.in_axes,
            out_axes=self.out_axes,
            axis_name=self.axis_name,
            axis_size=self.axis_size,
            spmd_axis_name=self.spmd_axis_name,
        )(*args, **kwargs)


class Partial(Module, Generic[T]):
    """
    Fix parts of the arguments of a callable module. Can be used in a similar
    manner to `functools.partial`.

    Parameters
    ---
    module: Callable[..., T]
        The module to partially apply.

    args: Any
        Arguments to fix.

    kwargs: Any
        Keyword arguments to fix.
    """

    def __init__(
        self,
        module: Callable[..., T],
        *args,
        **kwargs,
    ):
        self.func = module
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs) -> T:
        return self.func(*self.args, *args, **{**self.kwargs, **kwargs})
