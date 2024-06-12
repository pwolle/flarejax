import dataclasses
import functools
from types import MappingProxyType
from typing import Callable, Mapping, Self, Sequence, TypeVar

import jax

from ._module import Module
from ._typecheck import typecheck

__all__ = [
    "Jit",
    "VMap",
    "Partial",
]

M = TypeVar("M")


@jax.jit
def _jit_apply_layer(layer, *args, **kwargs):
    return layer(*args, **kwargs)


@typecheck
class Jit(Module):
    """
    Wrap a module into a Jitted module, that can be used in Jax transformations.

    Attributes:
    ---
    module: Module
        The module to wrap into a Jitted module.
    """

    module: Callable

    def __post_init__(self: Self) -> None:
        if not callable(self.module):
            error = f"Module {self.module} is not callable."
            raise ValueError(error)

    def __call__(self: Self, *args, **kwargs):
        return _jit_apply_layer(self.module, *args, **kwargs)


@typecheck
class VMap(Module):
    """
    A wrapper of 'jax.vmap', that returns a module, such that is compatible with
    module functionalities like serialization and the Jax Pytree utilities.

    Attributes:
    ---
    module: Module
        The module to apply the 'jax.vmap' to.

    in_axes: int | None
        The in_axes argument of 'jax.vmap'.

    out_axes: int | None
        The out_axes argument of 'jax.vmap'.
    """

    module: Module
    in_axes: int | None = 0
    out_axes: int | None = 0

    def __post_init__(self: Self) -> None:
        if not callable(self.module):
            error = f"Module {self.module} is not callable."
            raise ValueError(error)

    def __call__(self: Self, *args, **kwargs):
        assert callable(self.module)
        return jax.vmap(
            self.module,
            self.in_axes,
            self.out_axes,
        )(*args, **kwargs)


@typecheck
class Partial(Module):
    """
    A wrapper of 'functools.partial', that returns a module, such that is compatible
    with module functionalities like serialization and the Jax Pytree utilities.

    Attributes:
    ---
    module: callable
        The function to wrap into a partial function.

    *args: tuple
        The positional arguments to pass to the function.

    **kwargs: dict
        The keyword arguments to pass to the function.
    """

    module: Module
    args: Sequence = ()
    kwargs: Mapping = dataclasses.field(
        default_factory=lambda: MappingProxyType({}),
    )

    @classmethod
    def init(cls, module: Module, *args, **kwargs) -> Self:
        return cls(
            module=module,
            args=tuple(args),
            kwargs=MappingProxyType(kwargs),
        )

    def __post_init__(self: Self) -> None:
        if not callable(self.module):
            error = f"Function {self.module} is not callable."
            raise ValueError(error)

    def __call__(self: Self, *args, **kwargs):
        assert callable(self.module)
        return functools.partial(self.module, *self.args, **self.kwargs)(
            *args,
            **kwargs,
        )
