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

from ._tcheck import typecheck
from ._utils import array_str


__all__ = [
    "Module",
    "flatten",
    "unflatten",
]


@typecheck
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


@typecheck
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


@typecheck
class Module(PyTreeBase):
    def tree_flatten(self: Self):
        leaves, struct = flatten(self)

        active = []
        active_keys = []

        static = {}

        for key, val in leaves.items():
            if isinstance(val, jax.Array):
                active.append(val)
                active_keys.append(key)
                continue

            static[key] = val

        return active, (AuxData(struct, static, active_keys),)

    @classmethod
    def tree_unflatten(cls, aux_data, active):
        aux_data = aux_data[0]
        active = dict(zip(aux_data.active_keys, active))
        return unflatten(
            aux_data.static | active,
            aux_data.struct,
        )

    def __repr__(self) -> str:
        head = f"{type(self).__name__}("
        body = []

        for lookup in get_lookups(self):
            val = get_with_lookup(self, lookup)

            if not isinstance(val, Module):
                continue

            rep = repr(val)
            rep = f"{lookup.key}={rep}"

            body.append(rep)

        if not body:
            return head + ")"

        body = ",\n".join(body)
        body = "\n" + body

        # add indentation
        body = body.replace("\n", "\n  ")
        body = body + ",\n"

        tail = ")"
        return head + body + tail


NestedType: TypeAlias = Module | list | tuple | dict


@typecheck
@dataclasses.dataclass(frozen=True)
class ItemLookup:
    key: Hashable | int
    src: type | None

    def __repr__(self) -> str:
        return f"[{self.key}]"


@typecheck
@dataclasses.dataclass(frozen=True)
class AttrLookup:
    key: str
    src: type | None

    def __repr__(self) -> str:
        return f".{self.key}"


@typecheck
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

    if isinstance(obj, Module):
        keys = []

        if hasattr(obj, "__dict__"):
            keys.extend(sorted(obj.__dict__.keys()))

        if hasattr(obj, "__slots__"):
            keys.extend(sorted(obj.__slots__))  # type: ignore

        return [AttrLookup(k, src) for k in keys]

    raise ValueError(f"Unhashable type: {src}")


@typecheck
def get_with_lookup(
    obj: Any,
    /,
    lookup: ItemLookup | AttrLookup,
) -> Any:
    if isinstance(lookup, ItemLookup):
        return obj[lookup.key]

    return getattr(obj, lookup.key)


@typecheck
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
        assert len(obj) > lookup.key, "Index out of range"

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


@typecheck
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


@typecheck
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
            val = dfs(val, PathLookup((*path, lookup)))
            obj_ = set_with_lookup(obj_, lookup, val)

        return obj_

    return dfs(obj, PathLookup(()))


@typecheck
def get_lookup_deep(
    obj: Any,
    /,
    lookup_path: PathLookup,
) -> Any:
    for item in lookup_path.lookups:
        obj = get_with_lookup(obj, item)

    return obj


@typecheck
def dfs_map(obj: Any, fun: Callable[[PathLookup, Any], Any], /):
    obj = deref(obj)

    def dfs(obj_: Any, path: PathLookup):
        if isinstance(obj_, PathLookup):
            return obj_

        obj_ = copy.copy(obj_)

        for lookup in get_lookups(obj_):
            val = get_with_lookup(obj_, lookup)
            val = dfs(val, PathLookup((*path, lookup)))
            obj_ = set_with_lookup(obj_, lookup, val)

        obj_ = fun(path, obj_)
        return obj_

    obj = dfs(obj, PathLookup(()))
    return reref(obj)


@typecheck
def dfs_copy(obj_, /):
    if isinstance(obj_, PathLookup):
        return obj_

    obj_ = copy.copy(obj_)

    for lookup in get_lookups(obj_):
        val = get_with_lookup(obj_, lookup)
        val = dfs_copy(val)
        obj_ = set_with_lookup(obj_, lookup, val)

    return obj_


@typecheck
def reref(obj: Any, /):
    # can perform a deep copy, since there are no reference cycles
    obj = dfs_copy(obj)

    def dfs(obj_, path: PathLookup):
        if isinstance(obj_, PathLookup):
            return get_lookup_deep(obj, obj_)

        for lookup in get_lookups(obj_):
            val = get_with_lookup(obj_, lookup)
            val = dfs(val, PathLookup((*path, lookup)))
            obj_ = set_with_lookup(obj_, lookup, val)

        return obj_

    return dfs(obj, PathLookup(()))


SLOT = None


@typecheck
def flatten(obj: Any, /):
    obj = deref(obj)
    leaves: dict[PathLookup, Any] = {}

    def dfs(obj_: Any, /, path: PathLookup):
        if isinstance(obj_, PathLookup):
            return obj_

        if not get_lookups(obj_):
            leaves[path] = obj_
            return SLOT

        for lookup in get_lookups(obj_):
            val = get_with_lookup(obj_, lookup)
            val = dfs(val, PathLookup((*path, lookup)))
            obj_ = set_with_lookup(obj_, lookup, val)

        return obj_

    return leaves, dfs(obj, PathLookup(()))


@typecheck
def unflatten(leaves, obj: Any, /):
    obj = dfs_copy(obj)

    def dfs(obj: Any, path: PathLookup):
        if isinstance(obj, PathLookup):
            return obj

        if obj is SLOT:
            return leaves[path]

        for lookup in get_lookups(obj):
            val = get_with_lookup(obj, lookup)
            val = dfs(val, PathLookup((*path, lookup)))
            obj = set_with_lookup(obj, lookup, val)

        return obj

    obj = dfs(obj, PathLookup(()))
    return reref(obj)


@typecheck
def _attempt_hash(obj: Any, /) -> int:
    lookups = get_lookups(obj)

    if len(lookups) == 0:
        return hash(obj)

    hashes: list[Hashable] = [hash(type(obj))]

    for lookup in lookups:
        val = get_with_lookup(obj, lookup)
        hashes.append((hash(lookup), _attempt_hash(val)))

    return hash(tuple(hashes))


@typecheck
@dataclasses.dataclass(frozen=True)
class AuxData:
    struct: Any
    static: Any
    active_keys: Any

    def __hash__(self) -> int:
        return _attempt_hash(self)
