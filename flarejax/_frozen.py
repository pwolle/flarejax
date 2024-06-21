import dataclasses
from types import MappingProxyType
from typing import (
    Generic,
    Hashable,
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
    Self,
    Sequence,
    TypeVar,
    ValuesView,
    overload,
)

import jax.tree_util as jtu

from ._module import Module
from ._typecheck import typecheck

__all__ = [
    "ModuleMapping",
    "ModuleSequence",
    "Sequential",
    "SequenceKey",
]

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


@dataclasses.dataclass(frozen=True)
class SequenceKey:
    idx: int
    len: int | None = None  # to allow for matching with negative indices

    def __str__(self):
        return f"[{self.idx!r}]"


@typecheck
class ModuleSequence(Module, Generic[T]):
    _data: Sequence[T] = dataclasses.field(default_factory=lambda: ())

    def __post_init__(self: Self) -> None:
        # cast to tuple to make it harder to modify accidentally
        object.__setattr__(self, "_data", tuple(self._data))

    def __repr__(self: Self) -> str:
        head = f"{self.__class__.__name__}("
        body = []

        for value in self._data:
            body.append(f"{value!r}")

        body = ",\n".join(body)
        body = "\n" + body

        body = body.replace("\n", "\n  ")
        body = body + "\n"

        tail = ")"
        return head + body + tail

    @overload
    def __getitem__(self: Self, key: int, /) -> T: ...

    @overload
    def __getitem__(self: Self, key: slice, /) -> "ModuleSequence[T]": ...

    def __getitem__(self: Self, key: int | slice, /) -> T | "ModuleSequence[T]":
        if isinstance(key, slice):
            return ModuleSequence(self._data[key])

        return self._data[key]

    def __iter__(self: Self) -> Iterator[T]:
        return iter(self._data)

    def __len__(self: Self) -> int:
        return len(self._data)

    def __contains__(self: Self, value: T, /) -> bool:
        return value in self._data

    def __reverse__(self: Self) -> Self:
        return type(self)(tuple(reversed(self._data)))

    def __eq__(self: Self, other: object, /) -> bool:
        if not isinstance(other, ModuleSequence):
            return False

        return self._data == other._data

    def __ne__(self: Self, other: object, /) -> bool:
        return not self == other

    def index(self: Self, value: T, start: int = 0, stop: int = -1) -> int:
        return self._data.index(value, start, stop)

    def count(self: Self, value: T, /) -> int:
        return self._data.count(value)

    def append(self: Self, value: T, /) -> Self:
        return type(self)(tuple(self._data) + (value,))

    def extend(self: Self, values: Sequence[T], /) -> Self:
        return type(self)(tuple(self._data) + tuple(values))

    def insert(self: Self, index: int, value: T, /) -> Self:
        data = list(self._data)
        data.insert(index, value)
        return type(self)(data)

    def remove(self: Self, value: T, /) -> Self:
        data = list(self._data)
        data.remove(value)
        return type(self)(data)

    def pop(self: Self, index: int = -1, /) -> tuple[Self, T]:
        data = list(self._data)
        value = data.pop(index)
        return type(self)(data), value

    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[tuple[tuple[SequenceKey, T], ...], None]:
        children = []
        for i, v in enumerate(self):
            k = SequenceKey(i, len(self))
            children.append((k, v))

        return tuple(children), None

    @classmethod
    def tree_unflatten(cls, _, children: tuple[T, ...]) -> Self:
        return cls(children)


class Sequential(ModuleSequence):
    def __post_init__(self: Self) -> None:
        super().__post_init__()

        for module in self:
            if callable(module):
                continue

            error = f"Module {module} is not callable."
            raise ValueError(error)

    def __call__(self: Self, x):
        for module in self:
            assert callable(module), f"Module {module} is not callable."

            x = module(x)

        return x


@typecheck
class ModuleMapping(Module, Generic[K, T]):
    _data: Mapping[K, T] = dataclasses.field(
        default_factory=lambda: MappingProxyType({}),
    )

    def __post_init__(self: Self) -> None:
        object.__setattr__(self, "_data", MappingProxyType(self._data))

    def __repr__(self: Self) -> str:
        head = f"{self.__class__.__name__}({{"
        body = []

        for key in self.keys():
            value = self[key]
            body.append(f"{key!r}: {value!r}")

        body = ",\n".join(body)
        body = "\n" + body

        body = body.replace("\n", "\n  ")
        body = body + "\n"

        tail = "})"
        return head + body + tail

    def __getitem__(self: Self, key: K, /) -> T:
        return self._data[key]

    def __iter__(self: Self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self: Self) -> int:
        return len(self._data)

    def __contains__(self: Self, key: K, /) -> bool:
        return key in self._data

    def keys(self: Self) -> KeysView[K]:
        return self._data.keys()

    def values(self: Self) -> ValuesView[T]:
        return self._data.values()

    def items(self: Self) -> ItemsView[K, T]:
        return self._data.items()

    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[tuple[tuple[jtu.DictKey, T], ...], tuple[K, ...]]:
        children = []
        aux_data = tuple(sorted(self.keys(), key=hash))

        for key in aux_data:
            children.append((jtu.DictKey(key), self[key]))

        return tuple(children), aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[K, ...],
        children: tuple[T, ...],
    ) -> Self:
        data = {}
        for k, v in zip(aux_data, children):
            data[k] = v

        return cls(data)

    def pop(self: Self, key: K, /) -> tuple[Self, T]:
        data = dict(self._data)
        value = data.pop(key)
        return type(self)(data), value

    def update(self: Self, other: Mapping[K, T], /) -> Self:
        data = dict(self._data)
        data.update(other)
        return type(self)(data)
