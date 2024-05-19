import dataclasses
import functools
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

from ._typecheck import typecheck

__all__ = [
    "FrozenMapping",
    "FrozenMappingHashable",
    "FrozenSequence",
]

K = TypeVar("K", bound=Hashable)
T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


@typecheck
@dataclasses.dataclass(frozen=True)
class FrozenMapping(Generic[K, T]):
    _data: Mapping[K, T] = dataclasses.field(default_factory=lambda: {})

    def __post_init__(self: Self) -> None:
        # cast to MappingProxyType to make it harder to modify
        object.__setattr__(self, "_data", MappingProxyType(self._data))

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({dict(self._data)})"

    def __getitem__(self: Self, key: K) -> T:
        return self._data[key]

    def __iter__(self: Self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self: Self) -> int:
        return len(self._data)

    def __contains__(self: Self, key: K, /) -> bool:
        return key in self._data

    def __eq__(self: Self, other: object, /) -> bool:
        if not isinstance(other, FrozenMapping):
            return False

        return self._data == other._data

    def __ne__(self: Self, other: object, /) -> bool:
        return not self == other

    def keys(self: Self) -> KeysView[K]:
        return self._data.keys()

    def values(self: Self) -> ValuesView[T]:
        return self._data.values()

    def items(self: Self) -> ItemsView[K, T]:
        return self._data.items()

    def update(self: Self, other: Mapping[K, T], /) -> Self:
        return type(self)({**self._data, **other})

    def pop(self: Self, key: K, /, *d: T) -> tuple[Self, T]:
        if len(d) > 1:
            error = f"pop expected at most 2 arguments, {len(d) + 1} given"
            raise TypeError(error)

        _data = dict(self._data)
        value = _data.pop(key, *d)
        return type(self)(_data), value

    def get(self: Self, key: K, /, default: T) -> T:
        return self._data.get(key, default)

    def __frozen_set_item__(self: Self, key: Hashable, value: T) -> Self:
        return dataclasses.replace(self, _data={**self._data, key: value})

    def __frozen_del_item__(self: Self, key: K) -> Self:
        data = dict(self._data)
        del data[key]
        return dataclasses.replace(self, _data=data)


@typecheck
class FrozenMappingHashable(FrozenMapping[K, H]):
    def __post_init__(self: Self) -> None:
        # TODO maybe add nested coercion to hashable mapping and sequence
        super().__post_init__()

        if not all(isinstance(k, Hashable) for k in self):
            raise TypeError("All keys must be hashable")

        if not all(isinstance(v, Hashable) for v in self.values()):
            raise TypeError("All values must be hashable")

    # since hashing would be O(n) we cache the result
    @functools.cached_property
    def _hash(self: Self) -> int:
        unfolded = []

        for key in sorted(self, key=hash):
            unfolded.append((key, self[key]))

        return hash(tuple(unfolded))

    def __hash__(self: Self) -> int:
        return self._hash


@typecheck
@dataclasses.dataclass(frozen=True)
class FrozenSequence(Generic[T]):
    _data: Sequence[T] = dataclasses.field(default_factory=lambda: ())

    def __post_init__(self: Self) -> None:
        object.__setattr__(self, "_data", tuple(self._data))

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({list(self._data)})"

    @overload
    def __getitem__(self: Self, key: int, /) -> T: ...

    @overload
    def __getitem__(self: Self, key: slice, /) -> "FrozenSequence[T]": ...

    def __getitem__(self: Self, key: int | slice, /) -> T | "FrozenSequence[T]":
        if isinstance(key, slice):
            return FrozenSequence(self._data[key])

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
        if not isinstance(other, FrozenSequence):
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

    def __frozen_set_item__(self: Self, key: int, value: T) -> Self:
        data = list(self._data)
        data[key] = value  # type: ignore
        return dataclasses.replace(self, _data=data)

    def __frozen_del_item__(self: Self, key: int) -> Self:
        data = list(self._data)
        del data[key]
        return dataclasses.replace(self, _data=data)
