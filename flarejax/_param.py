import dataclasses
from typing import Generic, Self, TypeVar, Union

import jax.tree_util as jtree
from typeguard import typechecked

from . import _module

__all__ = [
    "Param",
]

T = TypeVar("T")


@typechecked
@dataclasses.dataclass
class Param(_module.Module, Generic[T]):
    value: T
    flags: set = dataclasses.field(default_factory=set)

    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[tuple[tuple[jtree.GetAttrKey, T]], set]:
        return ((jtree.GetAttrKey("value"), self.value),), self.flags

    @classmethod
    def tree_unflatten(
        cls, label: set, data: tuple[tuple[jtree.GetAttrKey, T]]
    ) -> Self:
        return cls(data[0], label)

    def __class_getitem__(cls, data_type) -> type:
        return type(cls.__name__, (cls,), {"data": data_type})
