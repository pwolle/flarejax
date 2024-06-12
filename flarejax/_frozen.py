import dataclasses
from typing import Generic, Iterator, Self, Sequence, TypeVar, overload

import jax.tree_util as jtu

from ._module import Module
from ._typecheck import typecheck

__all__ = [
    "ModuleSequence",
    "Sequential",
]

T = TypeVar("T")


@typecheck
class ModuleSequence(Module, Generic[T]):
    _data: Sequence[T] = dataclasses.field(default_factory=lambda: ())

    def __post_init__(self: Self) -> None:
        # cast to tuple to make it harder to modify accidentally
        object.__setattr__(self, "_data", tuple(self._data))

    def __repr__(self: Self) -> str:
        head = f"{self.__class__.__name__}("
        body = []

        for i, value in enumerate(self._data):
            body.append(f"{i}={value!r}")

        if sum(map(len, body)) > 60:
            body = ",\n".join(body)
            body = "\n" + body

            body = body.replace("\n", "\n  ")
            body = body + "\n"
        else:
            body = ", ".join(body)

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
    ) -> tuple[tuple[tuple[jtu.SequenceKey, T], ...], None]:
        children = []
        for i, v in enumerate(self):
            k = jtu.SequenceKey(i)
            children.append((k, v))

        return tuple(children), None

    @classmethod
    def tree_unflatten(cls, _, children: tuple[T, ...]) -> Self:
        return cls(children)


class Sequential(ModuleSequence[Module]):
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
