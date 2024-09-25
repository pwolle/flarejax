import dataclasses
from typing import Any, Callable, Generic, Hashable, ParamSpec, TypeVar

import jax

from .._filter import filter_jit
from .._module import Module

__all__ = [
    "Jit",
    "Vmap",
    "Partial",
]

T = TypeVar("T")
P = ParamSpec("P")


@filter_jit
def _filter_jit_apply(
    module: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    return module(*args, **kwargs)


@dataclasses.dataclass
class Jit(Module, Generic[P, T]):
    module: Callable[P, T]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return _filter_jit_apply(self.module, *args, **kwargs)


@dataclasses.dataclass
class Vmap(Module, Generic[P, T]):
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
