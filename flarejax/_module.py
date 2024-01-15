from typing import Any, Self

import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import typeguard

from . import _pytree

__all__ = [
    "Module",
    "ModuleList",
]


class Module(_pytree.PyTreeBase):
    """
    Base class for modules.
    """

    @typeguard.typechecked
    def tree_flatten(
        self: Self,
    ) -> tuple[tuple[dict[str, "Module"]], dict[str, Any]]:
        modules, statics = {}, {}

        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                modules[k] = v
            else:
                statics[k] = v

        return (modules,), statics

    @classmethod
    def tree_unflatten(
        cls: type[Self],
        statics: dict[str, Any],
        modules: tuple[dict[str, "Module"]],
    ) -> Self:
        instance = cls.__new__(cls)
        instance.__dict__ |= statics | modules[0]
        return instance

    @typeguard.typechecked
    def save_leaves(self: Self, path: str) -> None:
        leaves = jtree.tree_leaves(self)
        jnp.savez(path, *leaves)

    @typeguard.typechecked
    def load_leaves(
        self: Self,
        path: str,
        valid_dtype: bool = True,
        valid_shape: bool = True,
    ) -> Self:
        active_old, treedef = jtree.tree_flatten(self)

        active = jnp.load(path)  # type: ignore
        active_new = list(active.values())  # type: ignore

        for new, old in zip(active_new, active_old):
            if valid_dtype and new.dtype != old.dtype:
                error = f"Dtype: Loaded {new.dtype} != module {old.dtype}."
                raise ValueError(error)

            if valid_shape and new.shape != old.shape:
                error = f"Shape: Loaded {new.shape} != module {old.shape}."
                raise ValueError(error)

        return jtree.tree_unflatten(treedef, active_new)


class ModuleList(Module):
    """
    List of modules.
    """

    modules: list

    @typeguard.typechecked
    def __init__(self, *modules: Any) -> None:
        self.modules = list(modules)

    @typeguard.typechecked
    def tree_flatten(
        self: Self,
    ) -> tuple[tuple[dict[int, Module]], dict[int, Any]]:
        active = {}
        static = {}

        for i, value in enumerate(self.modules):
            if isinstance(value, Module):
                active[i] = value
            else:
                static[i] = value

        return (active,), static

    @classmethod
    def tree_unflatten(
        cls: type[Self],
        statics: dict[int, Any],
        modules: tuple[dict[int, Module]],
    ) -> Self:
        data = statics | modules[0]
        data = [data[i] for i in range(len(data))]
        return cls(*data)
