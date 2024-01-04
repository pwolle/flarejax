import dataclasses
from typing import Self, Generic, TypeVar

import typeguard

from . import _module


__all__ = [
    "Param",
]

T = TypeVar("T")


@dataclasses.dataclass
class Param(_module.Module, Generic[T]):
    data: T
    tags: dict = dataclasses.field(default_factory=dict)

    @typeguard.typechecked
    def tree_flatten(self: Self) -> tuple[tuple[T], dict]:
        return (self.data,), self.tags

    @classmethod
    def tree_unflatten(cls: type[Self], tags: dict, data: tuple[T]) -> Self:
        return cls(data[0], tags)

    def __class_getitem__(cls, data_type: type[T]) -> type:
        return type(cls.__name__, (cls,), {"data": data_type})
