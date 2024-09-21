import abc
import copy
import dataclasses
from pprint import pprint
from typing import Any, Hashable, Sequence, Self

import jax.tree_util as jtu


@dataclasses.dataclass(frozen=True)
class ItemLookup:
    key: Hashable | int

    def __repr__(self) -> str:
        return f"[{self.key}]"


@dataclasses.dataclass(frozen=True)
class AttrLookup:
    key: str

    def __repr__(self) -> str:
        return f".{self.key}"


@dataclasses.dataclass(frozen=True)
class PathLookup:
    path: Sequence[ItemLookup | AttrLookup | None] = ()

    def __repr__(self) -> str:
        if not self.path:
            return "obj"

        if self.path[-1] is None:
            s = repr(type(self)(self.path[:-1]))
            return f"type({s})"

        base = repr(type(self)())
        return base + "".join(map(repr, self.path))


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
    def tree_flatten(self, Self):
        pass


def get_lookups(
    obj: Module | list | tuple | dict | Any,
    /,
) -> list[ItemLookup | AttrLookup]:
    if not isinstance(obj, (Module, list, tuple, dict)):
        return []

    if isinstance(obj, (list, tuple)):
        return [ItemLookup(i) for i in range(len(obj))]

    if isinstance(obj, dict):
        return [ItemLookup(k) for k in sorted(obj.keys(), key=hash)]

    assert isinstance(obj, Module)

    keys = []

    if hasattr(obj, "__dict__"):
        for key in sorted(obj.__dict__.keys()):
            keys.append(AttrLookup(key))

    if hasattr(obj, "__slots__"):
        for key in sorted(obj.__slots__):  # type: ignore
            keys.append(AttrLookup(key))

    return keys


def get_with_lookup(
    obj: Module | list | tuple | dict,
    /,
    lookup: ItemLookup | AttrLookup,
) -> Any:
    if isinstance(lookup, AttrLookup):
        assert isinstance(lookup.key, str)
        return getattr(obj, lookup.key)

    assert isinstance(lookup, ItemLookup)
    return obj[lookup.key]  # type: ignore


def set_with_lookup(
    obj: Module | list | tuple | dict,
    lookup: ItemLookup | AttrLookup,
    value: Any,
) -> Any:
    if isinstance(lookup, AttrLookup):
        object.__setattr__(obj, lookup.key, value)
        return obj

    assert not isinstance(lookup, Module)

    if isinstance(obj, list):
        assert isinstance(lookup.key, int), "Expected `int` key"

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


def deref(obj: Any, /) -> Any:
    paths: dict[int, PathLookup] = {}

    def dfs(obj_, /, path: PathLookup):
        if id(obj_) in paths:
            return paths[id(obj_)]

        paths[id(obj_)] = path

        # perform shallow copy before modyfing the object
        obj_ = copy.copy(obj_)

        for lookup in get_lookups(obj_):
            val = get_with_lookup(obj_, lookup)
            val = dfs(val, PathLookup((*path.path, lookup)))
            obj_ = set_with_lookup(obj_, lookup, val)

        return obj_

    return dfs(obj, PathLookup(()))


def as_dicts(obj, path: PathLookup = PathLookup()):
    lookups = get_lookups(obj)

    if not lookups:
        return obj

    children: dict[ItemLookup | AttrLookup | None, Any]
    children = {None: type(obj)}

    for lookup in lookups:
        path_new = PathLookup((*path.path, lookup))

        child = get_with_lookup(obj, lookup)
        child = as_dicts(child, path_new)

        children[lookup] = child

    return children


def flatten_dict(nested_dict, path: tuple = ()):
    items = {}

    for key, value in nested_dict.items():
        path_new = path + (key,)

        if isinstance(value, dict):
            items.update(flatten_dict(value, path_new))
            continue

        items[path_new] = value

    return dict(items)


def split_dict(obj):
    leaves = {}
    etypes = {}

    for key, value in obj.items():
        if key and key[-1] is None:
            etypes[PathLookup(key[:-1])] = value
            continue

        leaves[PathLookup(key)] = value

    return leaves, etypes


def unflatten_dict(flat_dict):
    nested_dict = {}
    for path, value in flat_dict.items():
        current = nested_dict

        for key in path.path[:-1]:
            current = current.setdefault(key, {})

        current[path.path[-1]] = value

    return nested_dict


def main():
    obj = [1, 2, [3, 4], 5, [6, 7]]
    obj.append(obj)
    obj[2].append(obj[4])
    obj[4].append(obj[2])

    derefed = deref(obj)
    dicts = as_dicts(derefed)
    flatten = flatten_dict(dicts)
    unflat = unflatten_dict(flatten)

    print(obj)
    print()

    pprint(derefed)
    print()

    pprint(dicts)
    print()

    pprint(flatten)
    print()

    pprint(unflat)
    print()


if __name__ == "__main__":
    main()
