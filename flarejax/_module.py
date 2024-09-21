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
        flat = flatten(self)

        active = []
        acitve_keys = []

        static = []

        for key in sorted(flat.keys(), key=hash):
            value = flat[key]

            if isinstance(value, jax.Array):
                active.append(value)
                acitve_keys.append(key)
                continue

            static.append((key, value))

        return tuple(active), (tuple(static), tuple(acitve_keys))

    @classmethod
    def tree_unflatten(cls, aux_data, active):
        static, active_keys = aux_data
        static = dict(static)
        active = dict(zip(active_keys, active))
        return unflatten(static | active)

    def __repr__(self) -> str:
        return summary(self)


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
    path: Sequence[ItemLookup | AttrLookup]

    def __repr__(self) -> str:
        return "obj" + "".join(map(str, self.path))


TYPE_KEY = AttrLookup("__class__")


def object_to_dicts(
    obj: Any,
    /,
    path: PathLookup | None = None,
    refs: dict[int, PathLookup] | None = None,
):
    path = path or PathLookup(())
    refs = refs or {}

    if id(obj) in refs:
        return refs[id(obj)]

    if not isinstance(obj, (list, tuple, dict, Module)):
        return obj

    refs[id(obj)] = path

    obj_dict = {}
    obj_dict[TYPE_KEY] = type(obj)

    if isinstance(obj, dict):
        for key in sorted(obj.keys(), key=hash):
            key_path = PathLookup((*path.path, ItemLookup(key)))
            obj_dict[ItemLookup(key)] = object_to_dicts(obj[key], key_path, refs)

        return obj_dict

    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            key_path = PathLookup((*path.path, ItemLookup(i)))
            obj_dict[ItemLookup(i)] = object_to_dicts(value, key_path, refs)

        return obj_dict

    if isinstance(obj, Module):
        keys = []

        if hasattr(obj, "__dict__"):
            keys.extend(sorted(obj.__dict__.keys()))

        if hasattr(obj, "__slots__"):
            keys.extend(sorted(obj.__slots__))  # type: ignore

        for key in keys:
            key_path = PathLookup((*path.path, AttrLookup(key)))

            val = getattr(obj, key)
            obj_dict[AttrLookup(key)] = object_to_dicts(val, key_path, refs)

        return obj_dict

    return obj


def dicts_to_object(
    dicts: Any,
    /,
    path: PathLookup | None = None,
    refs: dict[PathLookup, Any] | None = None,
):
    path = path or PathLookup(())
    refs = refs or {}

    if isinstance(dicts, PathLookup):
        if dicts in refs:
            return refs[dicts]

        return dicts

    if not isinstance(dicts, dict):
        return dicts

    # shallow copy to avoid modifying the original through the pop
    dicts = copy.copy(dicts)
    cls = dicts.pop(TYPE_KEY)

    if cls is dict:
        result = {}
        refs[path] = result

        for key, value in dicts.items():
            key_path = PathLookup((*path.path, key))
            result[key.key] = dicts_to_object(value, key_path, refs)

        return result

    if cls is list:
        keys = sorted(dicts.keys(), key=lambda key: key.key)
        result = []
        refs[path] = result

        for key in keys:
            key_path = PathLookup((*path.path, key))
            result.append(dicts_to_object(dicts[key], key_path, refs))

        return result

    if cls is tuple:
        keys = sorted(dicts.keys(), key=lambda key: key.key)
        result = []

        for key in keys:
            key_path = PathLookup((*path.path, key))
            result.append(dicts_to_object(dicts[key], key_path, refs))

        result = tuple(result)

        # can only cache this after the the tuple is created
        # this is sound because a tuple cannot contain a reference to itself
        # by construction, because of it immutable nature
        refs[path] = result
        return result

    if issubclass(cls, Module):
        new = cls.__new__(cls)
        refs[path] = new

        for key, value in dicts.items():
            key_path = PathLookup((*path.path, key))
            val = dicts_to_object(value, key_path, refs)
            object.__setattr__(new, key.key, val)

        return new

    raise ValueError(f"Cannot reconstruct `{cls}`")


def dict_to_tuples(d: dict) -> tuple[tuple[Hashable, Any], ...]:
    keys = sorted(d.keys(), key=hash)
    return tuple((k, d[k]) for k in keys)


def tuples_to_dict(t: tuple) -> dict[Hashable, Any]:
    return {k: v for k, v in t}


def flatten_dict(
    nested_dict: dict,
    path: PathLookup = PathLookup(()),
) -> dict[PathLookup, Any]:
    items = {}

    for key, value in nested_dict.items():
        new_path = PathLookup((*path.path, key))

        if isinstance(value, dict):
            items.update(flatten_dict(value, new_path))
            continue

        items[new_path] = value

    return dict(items)


def unflatten_dict(flat_dict: dict[PathLookup, Any]) -> dict:
    nested_dict = {}

    for path, value in flat_dict.items():
        current = nested_dict

        for key in path.path[:-1]:
            current = current.setdefault(key, {})

        current[path.path[-1]] = value

    return nested_dict


def flatten(obj: Any) -> dict[PathLookup, Any]:
    obj_dict = object_to_dicts(obj)
    obj_flat = flatten_dict(obj_dict)  # type: ignore
    return obj_flat


def unflatten(obj_flat: dict[PathLookup, Any]) -> Any:
    obj_deep = unflatten_dict(obj_flat)
    obj_reco = dicts_to_object(obj_deep)
    return obj_reco


def array_summary(x: jax.Array, /) -> str:
    dtype = x.dtype.str[1:]
    shape = list(x.shape)

    head = f"{dtype}{shape}"
    return head


def _build_summary(head, body, tail):
    body = ",\n".join(body)
    body = body.replace("\n", "\n  ")

    if body:
        body = f"\n  {body},\n"

    return head + body + tail


def summary(obj: Any, /) -> str:
    if not isinstance(obj, (Module, tuple, list, dict, jax.Array)):
        return str(obj)

    if isinstance(obj, jax.Array):
        return array_summary(obj)

    if isinstance(obj, list):
        body = [summary(value) for value in obj]
        return _build_summary("[", body, "]")

    if isinstance(obj, tuple):
        body = [summary(value) for value in obj]
        return _build_summary("(", body, ")")

    if isinstance(obj, dict):
        body = [f"{key}: {summary(value)}" for key, value in obj.items()]
        return _build_summary("{", body, "}")

    assert isinstance(obj, Module)

    keys = []
    if hasattr(obj, "__slots__"):
        keys.extend(obj.__slots__)  # type: ignore

    if hasattr(obj, "__dict__"):
        keys.extend(obj.__dict__.keys())

    keys = filter(lambda key: not key.startswith("__"), keys)

    body = [f"{key}={summary(getattr(obj, key))}" for key in keys]
    return _build_summary(f"{type(obj).__name__}(", body, ")")
