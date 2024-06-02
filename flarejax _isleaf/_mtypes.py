import functools
from typing import Any, Callable, Generic, Self, TypeVar

import jax

from ._config import ConfigMapping, ConfigSequence, coerce_config_type
from ._frozen import FrozenMapping, FrozenSequence
from ._module import FieldKey, Module, field, is_leaf
from ._typecheck import typecheck

__all__ = [
    "ModuleMapping",
    "ModuleSequence",
    "Sequential",
    "Jit",
    "VMap",
    "Partial",
]

M = TypeVar("M")


@typecheck
class ModuleMapping(FrozenMapping[str, M], Module, Generic[M]):
    def __post_init__(self: Self) -> None:
        _data = dict(self._data)

        for k, v in _data.items():
            if is_leaf(v):
                continue

            _data[k] = coerce_config_type(v)  # type: ignore

        object.__setattr__(self, "_data", _data)
        super().__post_init__()

        if "_leaves" in self:
            error = "Key '_leaves' is reserved for internal use."
            raise ValueError(error)

        for k, v in self.items():
            if not isinstance(k, str):
                error = f"All keys must be strings, got {k}"
                raise TypeError(error)

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({dict(self._data)})"

    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[tuple[tuple[FieldKey, Any], ...], ConfigMapping]:
        items_active = []
        items_static = {}

        keys_leaves = []
        keys_module = sorted(self.keys())

        for module_key in keys_module:
            value = self[module_key]

            if is_leaf(value):
                keys_leaves.append(module_key)
                continue

            items_static[module_key] = value

        items_static["_leaves"] = ConfigSequence(keys_leaves)
        items_static = ConfigMapping(items_static)

        for module_key in keys_leaves:
            value = self[module_key]
            assert is_leaf(value)

            tree_key = FieldKey(
                name=module_key,
                type=type(self),
                tree=items_static,
            )
            items_active.append((tree_key, value))

        return tuple(items_active), items_static

    @classmethod
    def tree_unflatten(
        cls,
        static: ConfigMapping,
        active: tuple[M, ...],
    ) -> Self:
        static, leaves = static.pop("_leaves")

        active_items: dict[str, M] = {}
        assert isinstance(leaves, ConfigSequence)

        for i, k in enumerate(leaves):
            active_items[k] = active[i]

        return cls({**active_items, **static})


@typecheck
class ModuleSequence(FrozenSequence[M], Module, Generic[M]):
    def __post_init__(self: Self) -> None:
        _data = list(self._data)

        for i, v in enumerate(_data):
            if is_leaf(v):
                continue

            _data[i] = coerce_config_type(v)  # type: ignore

        object.__setattr__(self, "_data", _data)
        super().__post_init__()

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({list(self._data)})"

    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[tuple[tuple[FieldKey, Module], ...], ConfigMapping]:
        items_active = []
        items_static = {}

        for i, v in enumerate(self):
            if is_leaf(v):
                continue

            items_static[str(i)] = v

        items_static = ConfigMapping(items_static)

        for v in self:
            if not is_leaf(v):
                continue

            key = FieldKey(
                name="_data",
                type=type(self),
                tree=items_static,
            )
            items_active.append((key, v))

        return tuple(items_active), items_static

    @classmethod
    def tree_unflatten(
        cls,
        static: ConfigMapping,
        active: tuple[M, ...],
    ) -> Self:
        active_index = 0
        values = []

        for i in range(len(static) + len(active)):
            i = str(i)

            if i in static:
                values.append(static[i])

            else:
                values.append(active[active_index])
                active_index += 1

        return cls(values)


@typecheck
class Sequential(ModuleSequence[M], Generic[M]):
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
    in_axes: int | None = field(default=0)
    out_axes: int | None = field(default=0)

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
    args: ModuleSequence = field(default=ModuleSequence([]))
    kwargs: ModuleMapping = field(default=ModuleMapping({}))

    @classmethod
    def init(cls, module: Module, *args, **kwargs) -> Self:
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
