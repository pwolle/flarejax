import dataclasses
import functools
from typing import Callable, Hashable, Mapping, Self

import jax

from ._config import ConfigMapping, valid_conifg_types
from ._frozen import FrozenMapping, FrozenMappingHashable, FrozenSequence
from ._module import FieldKey, Module, field

__all__ = [
    "ModuleMapping",
    "ModuleSequence",
    "Sequential",
    "Jit",
    "VMap",
    "Partial",
]


class ModuleMapping(FrozenMapping[str, Module | Hashable], Module):
    def __post_init__(self: Self) -> None:
        super().__post_init__()

        if not all(isinstance(key, str) for key in self):
            error = "Keys mus be valid strings."
            raise TypeError(error)

        for key, value in self.items():
            if isinstance(value, (Module, jax.Array)):
                continue

            if valid_conifg_types(value):
                continue

            error = (
                f"'{self.__class__.__name__}' values must be Modules, jax"
                f"arrays or config tyes, but got '{value}' for key '{key}'."
            )
            raise TypeError(error)

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({dict(self._data)})"

    def __frozen_set_item__(self: Self, key: Hashable, value: Module) -> Self:
        return dataclasses.replace(self, _data={**self._data, key: value})

    def __frozen_del_item__(self: Self, key: str) -> Self:
        data = dict(self._data)
        del data[key]
        return dataclasses.replace(self, _data=data)

    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[
        tuple[tuple[FieldKey, dict[Hashable, Module]]],
        ConfigMapping,
    ]:
        active = {}
        static = {}

        for key, value in self.items():
            if isinstance(value, (Module, jax.Array)):
                active[key] = value
            else:
                static[key] = value

        active_key = FieldKey(
            name="_data",
            hint=type(self),
            grad=True,
            leaf=False,
            attr=False,
            meta=FrozenMappingHashable({}),
            conf=self.config,
        )
        return ((active_key, active),), ConfigMapping(static)

    @classmethod
    def tree_unflatten(
        cls: type[Self],
        static: Mapping,
        active: Mapping,
    ) -> Self:
        print(static, active)
        return cls({**static, **active[0]})


class ModuleSequence(FrozenSequence[Module | Hashable], Module):
    def __post_init__(self: Self) -> None:
        super().__post_init__()

        for value in self:
            if isinstance(value, (Module, jax.Array)):
                continue

            if valid_conifg_types(value):
                continue

            error = (
                f"'{self.__class__.__name__}' values must be Modules, jax"
                f"arrays or config tyes, but got '{value}'."
            )
            raise TypeError(error)

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({list(self._data)})"

    def __frozen_set_item__(self: Self, key: int, value: Module) -> Self:
        data = list(self._data)
        data[key] = value
        return dataclasses.replace(self, _data=data)

    def __frozen_del_item__(self: Self, key: int) -> Self:
        data = list(self._data)
        del data[key]
        return dataclasses.replace(self, _data=data)

    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[tuple[tuple[FieldKey, list[Module]]], ConfigMapping]:
        active = []
        static = {}

        for i, value in enumerate(self):
            if isinstance(value, (Module, jax.Array)):
                active.append(value)
            else:
                static[i] = value

        active_key = FieldKey(
            name="_data",
            hint=type(self),
            grad=True,
            leaf=False,
            attr=False,
            meta=FrozenMappingHashable({}),
            conf=self.config,
        )
        return ((active_key, active),), ConfigMapping(static)

    @classmethod
    def tree_unflatten(
        cls: type[Self],
        static: list[Hashable],
        active: tuple[list[Module]],
    ) -> Self:
        active_index = 0
        static_index = 0
        values = []

        for i in range(len(static) + len(active[0])):
            if i in static:
                values.append(static[static_index])
                static_index += 1

            else:
                values.append(active[0][active_index])
                active_index += 1

        return cls(values)


class Sequential(ModuleSequence):
    def __post_init__(self: Self) -> None:
        super().__post_init__()

        if not all(callable(layer) for layer in self):
            error = "All layers must be callable."
            raise ValueError(error)

    def __call__(self: Self, x):
        for layer in self:
            assert callable(layer)
            x = layer(x)

        return x


@jax.jit
def _jit_apply_layer(layer, *args, **kwargs):
    return layer(*args, **kwargs)


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
    in_axes: int | None = field(leaf=False, default=0)
    out_axes: int | None = field(leaf=False, default=0)

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
    args: ModuleSequence = field(leaf=False, default=ModuleSequence([]))
    kwargs: ModuleMapping = field(leaf=False, default=ModuleMapping({}))

    @classmethod
    def init(cls: type[Self], module, *args, **kwargs):
        return cls(
            module=module,
            args=ModuleSequence(args),
            kwargs=ModuleMapping(kwargs),
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
