import abc
import dataclasses
from typing import Any, Self, dataclass_transform

import jax.tree_util as jtu

from ._treemanip import ModuleAttr
from ._typecheck import typecheck


@typecheck
@dataclass_transform(frozen_default=True)
class FrozenDataclassMeta(abc.ABCMeta, type):
    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, Any],
        **_: Any,
    ):
        cls_new = super().__new__(cls, name, bases, attrs)
        cls_new = dataclasses.dataclass(frozen=True)(cls_new)
        return cls_new

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, Any],
        **_: Any,
    ) -> None:
        super().__init__(name, bases, attrs)
        jtu.register_pytree_with_keys_class(cls)


@typecheck
class FrozenDataclassBase(metaclass=FrozenDataclassMeta, register=False):
    @abc.abstractmethod
    def tree_flatten_with_keys(self: Self):
        error = "Abstract method 'tree_flatten_with_keys' must be implemented."
        raise NotImplementedError(error)

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux_data, children: tuple[Any, ...]) -> Self:
        error = "Abstract method 'tree_unflatten' must be implemented."
        raise NotImplementedError(error)


class Module(FrozenDataclassBase):
    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[tuple[tuple[ModuleAttr, Any], ...], None]:
        children = []
        selftype = type(self)

        for field in sorted(dataclasses.fields(self), key=lambda f: f.name):
            v = getattr(self, field.name)
            k = ModuleAttr(name=field.name, type=selftype, tree=None)
            children.append((k, v))

        return tuple(children), None

    @classmethod
    def tree_unflatten(cls, _, children: tuple[Any, ...]) -> Self:
        return cls(**{k.name: v for k, v in children})
