import abc
import copy
import dataclasses
from typing import (
    Any,
    Callable,
    Hashable,
    Self,
    Sequence,
    TypeAlias,
    overload,
)

import jax
import jax.tree_util as jtu


class PyTreeMeta(abc.ABCMeta, type):
    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, Any],
        **_: Any,
    ) -> None:
        super().__init__(name, bases, attrs)
        jtu.register_pytree_node_class(cls)


class PyTreeBase(metaclass=PyTreeMeta):
    @abc.abstractmethod
    def tree_flatten(self: Self):
        error = "Abstract method 'tree_flatten' must be implemented."
        raise NotImplementedError(error)

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux_data, children):
        error = "Abstract method 'tree_unflatten' must be implemented."
        raise NotImplementedError(error)


class Module(PyTreeBase):
    pass


@dataclasses.dataclass(frozen=True)
class ItemLookup:
    key: Hashable | int
    src: type | None

    def __repr__(self) -> str:
        return f"[{self.key}]"


@dataclasses.dataclass(frozen=True)
class AttrLookup:
    key: str
    src: type | None

    def __repr__(self) -> str:
        return f".{self.key}"


NestedType: TypeAlias = Module | list | tuple | dict


def get_lookups(obj: Any, /) -> list[ItemLookup | AttrLookup]:
    src = type(obj)

    if not isinstance(obj, NestedType):
        return []

    if isinstance(obj, (list, tuple)):
        return [ItemLookup(i, src) for i in range(len(obj))]

    if isinstance(obj, dict):
        return [ItemLookup(k, src) for k in sorted(obj.keys(), key=hash)]

    if isinstance(obj, set):
        return []

    assert isinstance(obj, Module)
    keys = []

    if hasattr(obj, "__dict__"):
        keys.extend(sorted(obj.__dict__.keys()))

    if hasattr(obj, "__slots__"):
        keys.extend(sorted(obj.__slots__))  # type: ignore

    return [AttrLookup(k, src) for k in keys]


NOTHING = object()


def set_with_lookup(
    obj: NestedType,
    lookup: ItemLookup | AttrLookup,
    value: Any,
) -> Any:
    if not isinstance(obj, NestedType):
        raise TypeError(f"Cannot set lookup on `{type(obj)}`")

    if isinstance(lookup, AttrLookup):
        object.__setattr__(obj, lookup.key, value)
        return obj

    if isinstance(obj, list):
        assert isinstance(lookup.key, int), "Expected `int` key"

        if len(obj) <= lookup.key:
            obj.extend([None] * (lookup.key - len(obj) + 1))

        obj[lookup.key] = value
        return obj

    if isinstance(obj, tuple):
        assert isinstance(lookup.key, int), "Expected `int` key"

        obj_mut = list(obj)
        obj_mut[lookup.key] = value

        return tuple(obj)

    if isinstance(obj, dict):
        obj[lookup.key] = value
        return obj

    raise TypeError(f"Cannot set lookup on `{type(obj)}`")


def get_with_lookup(
    obj: Any,
    /,
    lookup: ItemLookup | AttrLookup,
) -> Any:
    if isinstance(lookup, AttrLookup):
        return getattr(obj, lookup.key)

    assert isinstance(lookup, ItemLookup)

    if isinstance(obj, (list, tuple)):
        assert isinstance(lookup.key, int), "Expected `int` key"

        if lookup.key >= len(obj):
            return False

        print(obj, lookup.key, obj[lookup.key] is not NOTHING)
        return obj[lookup.key] is not NOTHING

    if isinstance(obj, dict):
        return lookup.key in obj

    raise TypeError(f"Cannot get lookup on `{type(obj)}`")


# def has_with_lookup(obj: Any, lookup: ItemLookup | AttrLookup) -> bool:
#     if isinstance(lookup, AttrLookup):
#         return hasattr(obj, lookup.key)


@dataclasses.dataclass(frozen=True)
class PathLookup:
    lookups: Sequence[ItemLookup | AttrLookup]

    @overload
    def __getitem__(self, key: int) -> ItemLookup | AttrLookup: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    def __getitem__(self, key: int | slice) -> Self | ItemLookup | AttrLookup:
        if isinstance(key, int):
            return self.lookups[key]

        assert isinstance(key, slice), "Expected `int` or `slice`"
        return type(self)(self.lookups[key])

    def __len__(self) -> int:
        return len(self.lookups)

    def __iter__(self):
        return iter(self.lookups)

    def __reversed__(self):
        return reversed(self.lookups)

    def __repr__(self) -> str:
        return "obj" + "".join(map(str, self.lookups))


def flatten(obj: Any):
    path_to_leaf: dict[PathLookup, Any] = {}
    path_from_id: dict[int, PathLookup] = {}

    def dfs(obj_, path: PathLookup):
        if id(obj_) in path_from_id:
            path_to_leaf[path] = path_from_id[id(obj_)]
            return

        path_from_id[id(obj_)] = path
        lookups = get_lookups(obj_)

        if not lookups:
            path_to_leaf[path] = obj_
            return

        for lookup in lookups:
            val = get_with_lookup(obj_, lookup)
            dfs(val, PathLookup((*path, lookup)))

    dfs(obj, PathLookup(()))
    return path_to_leaf


def unflatten(leaves):
    obj_type = next(iter(leaves))[0].src
    obj = obj_type.__new__(obj_type)


def main():
    from pprint import pprint

    obj = [1, 2, [3, 4], 5, [6, 7]]
    obj.append(obj)
    obj[2].append(obj[4])
    obj[4].append(obj[2])

    path_to_leaf = flatten(obj)

    print(obj)
    print()

    pprint(path_to_leaf)


if __name__ == "__main__":
    main()
