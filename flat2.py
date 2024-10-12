import dataclasses
from pprint import pprint
from typing import Any, Callable, Hashable


@dataclasses.dataclass(frozen=True)
class ItemLookup:
    """
    Describes how to lookup an item in a dictionary or list or tuple.
    """

    key: Hashable | int

    def __repr__(self) -> str:
        return f"[{self.key}]"


@dataclasses.dataclass(frozen=True)
class AttrLookup:
    """
    Describes how to lookup an attribute in a class.
    """

    key: str

    def __repr__(self) -> str:
        return f".{self.key}"


@dataclasses.dataclass(frozen=True)
class PathLookup:
    """
    Describes how to lookup a value in a nested structure.
    """

    path: tuple[ItemLookup | AttrLookup, ...]

    def __repr__(self) -> str:
        return "obj" + "".join(map(str, self.path))

    def __add__(self, other):
        if isinstance(other, PathLookup):
            return PathLookup(self.path + other.path)

        # assert isinstance(other, (ItemLookup, AttrLookup))
        if not isinstance(other, (ItemLookup, AttrLookup)):
            raise TypeError(f"Cannot add {type(other)} to Lookup.")

        return PathLookup(self.path + (other,))

    def __lt__(self, other):
        return hash(self) < hash(other)


@dataclasses.dataclass(frozen=True)
class RegistryEntry:
    get_with_lookup: Callable
    set_with_lookup: Callable
    get_all_lookups: Callable


REGISTRY = {}


def register(cls, entry: RegistryEntry):
    print(f"Registering {cls.__name__}")

    REGISTRY[cls] = entry
    return cls


def get_with_lookup(obj, lookup):
    return REGISTRY[type(obj)].get_with_lookup(obj, lookup)


def set_with_lookup(obj, lookups, value):
    return REGISTRY[type(obj)].set_with_lookup(obj, lookups, value)


def get_all_lookups(obj):
    return REGISTRY[type(obj)].get_all_lookups(obj)


def get_with_lookup_list(obj: list, lookup: ItemLookup) -> Any:
    assert isinstance(lookup.key, int)
    return obj[lookup.key]


def set_with_lookup_list(obj: list, lookups: ItemLookup, value: Any) -> None:
    assert isinstance(lookups.key, int)

    if lookups.key >= len(obj):
        obj.extend([None] * (lookups.key - len(obj) + 1))

    obj[lookups.key] = value


def get_all_lookups_list(obj: list) -> list[ItemLookup]:
    return [ItemLookup(i) for i in range(len(obj))]


register(
    list,
    RegistryEntry(
        get_with_lookup=get_with_lookup_list,
        set_with_lookup=set_with_lookup_list,
        get_all_lookups=get_all_lookups_list,
    ),
)


def get_with_lookup_dict(obj: dict, lookup: ItemLookup) -> Any:
    return obj[lookup.key]


def set_with_lookup_dict(obj: dict, lookups: ItemLookup, value: Any) -> None:
    obj[lookups.key] = value


def get_all_lookups_dict(obj: dict) -> list[ItemLookup]:
    return [ItemLookup(k) for k in sorted(obj.keys(), key=hash)]


register(
    dict,
    RegistryEntry(
        get_with_lookup=get_with_lookup_dict,
        set_with_lookup=set_with_lookup_dict,
        get_all_lookups=get_all_lookups_dict,
    ),
)


def get_with_lookup_module(obj, lookup: AttrLookup) -> Any:
    return getattr(obj, lookup.key)


def set_with_lookup_module(obj, lookups: AttrLookup, value: Any) -> None:
    object.__setattr__(obj, lookups.key, value)


def get_all_lookups_module(obj) -> list[AttrLookup]:
    keys = []

    if hasattr(obj, "__dict__"):
        keys.extend(obj.__dict__.keys())

    if hasattr(obj, "__slots__"):
        keys.extend(obj.__slots__)  # type: ignore

    keys = sorted(set(keys))
    return [AttrLookup(k) for k in keys]


def register_module(cls):
    register(
        cls,
        RegistryEntry(
            get_with_lookup=get_with_lookup_module,
            set_with_lookup=set_with_lookup_module,
            get_all_lookups=get_all_lookups_module,
        ),
    )


class ModuleMeta(type):
    def __init__(cls, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        register_module(cls)


class ModuleBase(metaclass=ModuleMeta):
    pass


class Module(ModuleBase):
    pass


# To be able to reconstruct any object from the flatttened dictionary
# representation we need to store the types of all of the nodes.
TYPE_KEY = AttrLookup("__class__")


def object_to_dicts(obj_):
    refs = {}

    def dfs(obj, path):
        if id(obj) in refs:
            return refs[id(obj)]

        if type(obj) not in REGISTRY:
            refs[id(obj)] = path
            return obj

        refs[id(obj)] = path

        obj_dict = {}
        obj_dict[TYPE_KEY] = type(obj)

        for key in get_all_lookups(obj):
            key_path = path + key
            obj_dict[key] = dfs(get_with_lookup(obj, key), key_path)

        return obj_dict

    return dfs(obj_, PathLookup(()))


def dicts_to_object(obj_):
    def dfs_tree(data):
        if not isinstance(data, dict):
            return data

        if TYPE_KEY not in data:
            return data

        obj_type = data.pop(TYPE_KEY)
        obj = obj_type.__new__(obj_type)

        for key, value in data.items():
            set_with_lookup(obj, key, dfs_tree(value))

        return obj

    obj_ = dfs_tree(obj_)

    def dfs_refs(obj):
        if isinstance(obj, PathLookup):
            val = obj_

            for key in obj.path:
                val = get_with_lookup(val, key)

            return val

        if type(obj) not in REGISTRY:
            return obj

        for key in get_all_lookups(obj):
            set_with_lookup(
                obj,
                key,
                dfs_refs(get_with_lookup(obj, key)),
            )

        return obj

    return dfs_refs(obj_)


def dict_to_tuples(d: dict) -> tuple[tuple[Hashable, Any], ...]:
    """
    Convert a dictionary to a tuple of key-value pairs, sorted by the hash of
    the keys.
    This is useful to create a hashable representation of a dictionary, i.e.
    for the auxiliary data in jax PyTree.

    Parameters
    ---
    d: dict
        Dictionary to convert.

    Returns
    ---
    tuple[tuple[Hashable, Any], ...]
        A n-tuple of key-value 2-tuples.
    """
    keys = sorted(d.keys(), key=hash)
    return tuple((k, d[k]) for k in keys)


@typecheck
def tuples_to_dict(t: tuple) -> dict[Hashable, Any]:
    """
    Inverse of the `dict_to_tuples` function.

    Parameters
    ---
    t: tuple
        N-tuple of key-value 2-tuples.

    Returns
    ---
    dict[Hashable, Any]
        Dictionary representation of the n-tuple.
    """
    return {k: v for k, v in t}


@typecheck
def flatten_dict(
    nested_dict: dict,
    path=(),
) -> dict:
    """
    Flatten a nested dictionary into a dictionary of n-tuple paths and values.

    Parameters
    ---
    nested_dict: dict
        Nested dictionary to flatten.

    Returns
    ---
    dict
        Flattened dictionary. Each key is a n-tuple path of the keys to the leaf
        value of the original dictionary.
    """

    items = {}

    for key, value in nested_dict.items():
        new_path = (*path, key)

        if isinstance(value, dict):
            items.update(flatten_dict(value, new_path))
            continue

        items[new_path] = value

    return dict(items)


@typecheck
def unflatten_dict(flat_dict: dict[tuple[Hashable, ...], Any]) -> dict:
    """
    Inverse of the `flatten_dict` function.

    Parameters
    ---
    flat_dict: dict
        Flattened dictionary to unflatten. The keys muse be n-tuples of
        dictionary keys.

    Returns
    ---
    dict
        A nested dictionary where each key is a part of the n-tuple path to the
        leaf value of the original dictionary.
    """

    nested_dict = {}

    for path, value in flat_dict.items():
        current = nested_dict

        for key in path[:-1]:
            if key not in current:
                current[key] = {}

            current = current[key]

        current[path[-1]] = value

    return nested_dict


def flatten(obj: Any) -> dict[PathLookup, Any]:
    """
    Flatten an object into a dictionary of paths and values.
    Values can be the leaves of the object, references to other parts of the
    original object, or information about the types of non-leaf nodes.

    This function is especially useful in combination with `unflatten` to
    reconstruct an object from the flattened representation.

    Parameters
    ---
    obj: Any
        Object to flatten.

    Returns
    ---
    dict[PathLookup, Any]
        Flattened object. The Pathlookup describes how to get the leaf value
        from the original object.
    """
    obj_dict = object_to_dicts(obj)

    assert isinstance(obj_dict, dict)
    obj_flat = flatten_dict(obj_dict)

    obj_path = {PathLookup(k): v for k, v in obj_flat.items()}
    return obj_path


a = [1, 2]
b = {"a": a, "b": a}
b["c"] = b

flat = object_to_dicts(b)
reco = dicts_to_object(flat)

pprint(flat)
pprint(reco)
