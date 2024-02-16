from typing import Any, Mapping, Self

import jax.numpy as jnp
import jax.tree_util as jtree
from typeguard import typechecked

from . import _pytree

__all__ = [
    "Module",
    "ModuleList",
]


@typechecked
class Module(_pytree.PyTreeBase):
    """
    Base class for modules.
    """

    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[tuple[tuple[jtree.GetAttrKey, "Module"], ...], dict[str, Any]]:
        modules = []
        statics = []

        module_keys = []
        static_keys = []

        for name in sorted(self.__dict__.keys()):
            value = self.__dict__[name]

            if isinstance(value, Module):
                modules.append((jtree.GetAttrKey(name), value))
                module_keys.append(name)
                continue

            statics.append(value)
            static_keys.append(name)

        children = tuple(modules)
        aux_data = {
            "module_keys": module_keys,
            "static_keys": static_keys,
            "statics": statics,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, Any], children: tuple["Module", ...]
    ) -> Self:
        modules = dict(zip(aux_data["module_keys"], children))
        statics = dict(zip(aux_data["static_keys"], aux_data["statics"]))

        instance = cls.__new__(cls)
        instance.__dict__ |= statics | modules
        return instance

    def save_leaves(self: Self, path: str) -> None:
        flat, _ = jtree.tree_flatten_with_path(self)

        leaves = {}
        for key_path, value in flat:
            leaves[f"leaves{jtree.keystr(key_path)}"] = value

        jnp.savez(path, **leaves)

    def load_leaves(self: Self, path: str):
        flat_old, treedef = jtree.tree_flatten_with_path(self)
        flat_new = []

        leaves: Mapping = jnp.load(path)  # type: ignore
        keys = set()

        for key_path, value_old in flat_old:
            key = f"leaves{jtree.keystr(key_path)}"
            keys.add(key)

            value_new = leaves[key]

            if value_new.dtype != value_old.dtype:
                error = f"Dtype: Loaded {value_new.dtype} != {value_old.dtype}"
                raise ValueError(error)

            if value_new.shape != value_old.shape:
                error = f"Shape: Loaded {value_new.shape} != {value_old.shape}"
                raise ValueError(error)

            flat_new.append((key_path, value_new))

        if keys != set(leaves.keys()):
            error = f"Keys: Loaded {set(leaves.keys())} != {keys}"
            raise ValueError(error)

        return jtree.tree_unflatten(treedef, flat_new)


@typechecked
class ModuleList(Module):
    """
    List of modules.
    """

    modules: list[Module | Any]

    def __init__(self, *modules: Any) -> None:
        self.modules = list(modules)

    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[tuple[tuple[jtree.SequenceKey, Module], ...], dict[str, list]]:
        modules = []
        statics = []

        module_indicies = []
        static_indicies = []

        for i, value in enumerate(self.modules):
            if isinstance(value, Module):
                modules.append((jtree.SequenceKey(i), value))
                module_indicies.append(i)
                continue

            statics.append(value)
            static_indicies.append(i)

        children = tuple(modules)
        aux_data = {
            "module_keys": module_indicies,
            "static_keys": static_indicies,
            "statics": statics,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, list], children: tuple["Module", ...]
    ) -> Self:
        length = max(
            max(aux_data["module_keys"], default=-1),
            max(aux_data["static_keys"], default=-1),
        )

        modules = dict(zip(aux_data["module_keys"], children))
        statics = dict(zip(aux_data["static_keys"], aux_data["statics"]))

        modules = modules | statics
        return cls(*[modules[i] for i in range(length + 1)])
