import abc
import copy
from typing import Any, Hashable, Self, Callable

import jax
import jax.tree_util as jtu

from ._lookup import AttrLookup, ItemLookup, PathLookup
from ._tcheck import typecheck

__all__ = [
    "Module",
    "flatten",
    "unflatten",
]


@typecheck
class PyTreeMeta(abc.ABCMeta, type):
    """
    Meta class that registers all its subclasses with JAX PyTrees.
    This class also wraps all methods of the subclass with a `MethodDescriptor`.
    """

    def __new__(cls, name, bases, dct):
        for key, value in dct.items():
            if name in ["PyTreeBase", "Module", "MethodWrap"]:
                continue

            if not callable(value):
                continue

            if key in ["tree_flatten", "tree_unflatten"]:
                continue

            if key.startswith("__"):
                continue

            dct[key] = MethodDescriptor(value)

        return super().__new__(cls, name, bases, dct)

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
    """
    Abstract base class for PyTree classes.

    The `tree_flatten` method for deconstructing and the `tree_unflatten` for
    reconstructing it must be implemented.
    """

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
    """
    Base class for all modules. Modules can be used to implement parameterized
    functions like neural networks.

    Modules are PyTrees, which means they can be flattened and unflattened
    using JAX's tree_util functions.

    The flattening will automatically go through all attributes including slots.
    Submodules as well as dictionaries, lists, and tuples will also be
    recursively flattened.

    The flattening and unflattening will respect references within the object,
    although circular references to tuples are not supported for performance
    reasons.

    See also `flatten` and `unflatten` to get a flattened representation in
    the form of a dictionary with descriptions of how to lookup the values as
    keys.
    """

    def tree_flatten(self: Self):
        """
        Convert the module into a tuple of its array valued leaves and
        auxiliary data.

        The auxillary data is a 2-tuple of the all other leaves with their keys
        and the keys of the array valued leaves.
        They keys are the paths to the leaves in the original module.
        """
        flat = flatten(self)

        active = []
        active_keys = []

        static = []

        for key in sorted(flat.keys(), key=hash):
            value = flat[key]

            if isinstance(value, jax.Array):
                active.append(value)
                active_keys.append(key)
                continue

            static.append((key, value))

        return tuple(active), (tuple(static), tuple(active_keys))

    @classmethod
    def tree_unflatten(cls, aux_data, active):
        """
        Reconstruct the module from its flattened representation.
        """
        static, active_keys = aux_data
        static = dict(static)
        active = dict(zip(active_keys, active))
        return unflatten(static | active)

    def __repr__(self) -> str:
        return summary(self)

    @property
    @typecheck
    def frozen(self) -> bool:
        if not hasattr(self, "_frozen"):
            self._frozen = False

        return self._frozen

    @frozen.setter
    @typecheck
    def frozen(self, value: bool):
        self._frozen = value


# To be able to reconstruct any object from the flatttened dictionary
# representation we need to store the types of all of the nodes.
TYPE_KEY = AttrLookup("__class__")


@typecheck
def object_to_dicts(
    obj: Any,
    /,
    path: PathLookup | None = None,
    refs: dict[int, PathLookup] | None = None,
):
    """
    Convert any object recursively into a tree of nested dictionaries of its
    children and their types.

    Shared references within the object will be represented by `PathLookup`
    objects as values that point to the first occurrence of the reference.

    Parameters
    ---
    obj: Any
        Object to convert to a dictionary.

    path: PathLookup, optional
        Path to the current object in the tree.

    refs: dict[int, PathLookup], optional
        References to already visited objects.

    Returns
    ---
    Any
        Nested dictionary representation of the object.
        If the object is a leaf the object ifself will be returned.
        if the object is in `refs` a `PathLookup` object will be returned.
    """
    path = path or PathLookup(())
    refs = refs or {}

    # check if the object is already in the references, i.e. has been visited
    if id(obj) in refs:
        return refs[id(obj)]

    # if the object is a leaf, return it
    if not isinstance(obj, (list, tuple, dict, Module)):
        return obj

    # store the path to the object in the references
    # i.e. mark the current object as visited
    refs[id(obj)] = path

    obj_dict = {}
    obj_dict[TYPE_KEY] = type(obj)  # save the type of the object

    # destructure the object
    if isinstance(obj, dict):
        # keep a deterministic order of the keys, since this might be relevant
        # in jax.jit recompilations; this can be done by sorting the keys
        # by their hashes since dict keys are not guaranteed to be hashable

        for key in sorted(obj.keys(), key=hash):
            # the path to the current child is one level deeper than the
            # path to the current object
            # key_path = Lookup((*path.path, ItemLookup(key)))
            key_path = path + ItemLookup(key)

            # recursively convert the value to a dictionary
            obj_dict[ItemLookup(key)] = object_to_dicts(
                obj[key],
                key_path,
                refs,
            )

        return obj_dict

    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            # key_path = Lookup((*path.path, ItemLookup(i)))
            key_path = path + ItemLookup(i)
            obj_dict[ItemLookup(i)] = object_to_dicts(value, key_path, refs)

        return obj_dict

    if isinstance(obj, Module):
        keys = []

        if hasattr(obj, "__dict__"):
            keys.extend(sorted(obj.__dict__.keys()))

        if hasattr(obj, "__slots__"):
            keys.extend(sorted(obj.__slots__))  # type: ignore

        for key in keys:
            # key_path = Lookup((*path.path, AttrLookup(key)))
            key_path = path + AttrLookup(key)

            val = getattr(obj, key)
            obj_dict[AttrLookup(key)] = object_to_dicts(val, key_path, refs)

        return obj_dict

    return obj


@typecheck
def dicts_to_object(
    dicts: Any,
    /,
    path: PathLookup | None = None,
    refs: dict[PathLookup, Any] | None = None,
):
    path = path or PathLookup(())
    refs = refs or {}

    if isinstance(dicts, PathLookup):
        # The reference might not exist, this can be the case if there is a
        # reference cycle where the reference would point to a tuple from
        # one of its children (or children's children, etc.)
        if dicts in refs:
            return refs[dicts]

        # In this case the best we can do is to return a placeholder, i.e.
        # the path to the reference itself
        return dicts

    # if it is not a dictionary, it must be a leaf and can be returned as is
    if not isinstance(dicts, dict):
        return dicts

    # shallow copy to avoid modifying the original through the pop
    dicts = copy.copy(dicts)
    cls = dicts.pop(TYPE_KEY)

    if cls is dict:
        result = {}

        # since the result is mutable we can already store it in the references
        # such that if a reference cycle comes up the object can simply be
        # retrieved from the references dictionary
        refs[path] = result

        for key, value in dicts.items():
            # the path to the current child is one level deeper than the
            # path to the current object
            # key_path = Lookup((*path.path, key))
            key_path = path + key

            # apply the conversion recursively
            result[key.key] = dicts_to_object(value, key_path, refs)

        return result

    if cls is list:
        keys = sorted(dicts.keys(), key=lambda key: key.key)
        result = []
        refs[path] = result

        for key in keys:
            # key_path = Lookup((*path.path, key))
            key_path = path + key
            result.append(dicts_to_object(dicts[key], key_path, refs))

        return result

    if cls is tuple:
        keys = sorted(dicts.keys(), key=lambda key: key.key)
        result = []

        for key in keys:
            # key_path = Lookup((*path.path, key))
            key_path = path + key
            result.append(dicts_to_object(dicts[key], key_path, refs))

        result = tuple(result)

        # The result can only be stored after the tuple is created,
        # because the tupe is immutable
        refs[path] = result
        return result

    if issubclass(cls, Module):
        new = cls.__new__(cls)
        refs[path] = new

        for key, value in dicts.items():
            # key_path = Lookup((*path.path, key))
            key_path = path + key
            val = dicts_to_object(value, key_path, refs)
            object.__setattr__(new, key.key, val)

        return new

    # should be unreachable for objects that were created with `object_to_dicts`
    error = f"Cannot reconstruct `{cls}`."
    raise ValueError(error)


@typecheck
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


@typecheck
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


@typecheck
def unflatten(obj_path: dict[PathLookup, Any]) -> Any:
    """
    Reconstruct an object that was flattened with `flatten`.

    Parameters
    ---
    obj_path: dict[PathLookup, Any]
        Flattened object.

    Returns
    ---
    Any
        Reconstructed object.
    """
    obj_flat: dict[tuple[Hashable, ...], Any]
    obj_flat = {k.path: v for k, v in obj_path.items()}

    obj_deep = unflatten_dict(obj_flat)
    obj_reco = dicts_to_object(obj_deep)
    return obj_reco


@typecheck
def array_summary(x: jax.Array, /) -> str:
    """
    Summarize an array by its dtype and shape.

    Parameters
    ---
    x: jax.Array
        Array to summarize.

    Returns
    ---
    str
        Summary of its dtype and shape.
    """
    dtype = x.dtype.str[1:]
    shape = list(x.shape)

    head = f"{dtype}{shape}"
    return head


@typecheck
def _build_summary(head: str, body: list[str], tail: str) -> str:
    """
    Helper function for building a summary of a nested object.
    """
    body_str = ",\n".join(body)
    body_str = body_str.replace("\n", "\n  ")

    if body:
        body_str = f"\n  {body_str},\n"

    return head + body_str + tail


@typecheck
def summary(obj: Any, /) -> str:
    """
    Convert an object to a human-readable string representation, by formatting
    it and replacing arrays by their shape and dtype.

    Parameters
    ---
    obj: Any
        Object to summarize.

    Returns
    ---
    str
        Human-readable string representation of the object.
    """
    if not isinstance(obj, (Module, tuple, list, dict, jax.Array)):
        return repr(obj)

    # compress arrays to their shape and dtype
    if isinstance(obj, jax.Array):
        return array_summary(obj)

    if isinstance(obj, list):
        # apply recursively to all elements
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

    keys = filter(lambda key: not key.startswith("_"), keys)

    body = [f"{key}={summary(getattr(obj, key))}" for key in keys]
    return _build_summary(f"{type(obj).__name__}(", body, ")")


class MethodWrap(Module):
    """
    Wrapper for calling a method of a module. This makes it possible to use
    the benefits of callable PyTrees for all of the methods of a module.
    """

    @typecheck
    def __init__(self, module: Module, method: Callable):
        self.module = module
        self.method = method

    @property
    def __name__(self):
        return self.method.__name__

    @property
    def __self__(self):
        return self.module

    def __call__(self, *args, **kwargs):
        return self.method(self.module, *args, **kwargs)


class MethodDescriptor:
    """
    Descriptor for making methods of a class callable PyTrees.
    """

    @typecheck
    def __init__(self, func: Callable):
        self.func = func

    def __get__(self, instance, _):
        if instance is None:
            return self.func

        return MethodWrap(instance, self.func)
