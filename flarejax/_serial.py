import json

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as onp

from ._module import (
    AttrLookup,
    ItemLookup,
    MethodWrap,
    Module,
    PathLookup,
    flatten,
    unflatten,
)

__all__ = [
    "saveable",
    "save",
    "load",
]


_SAVEABLE_TYPES: dict[str, type] = {}
_SAVEABLE_TYPES_INV: dict[type, str] = {}


def saveable(name, warn: bool = True):
    """
    Store an object in a global registry, such that it can be serialized and
    deserialized by the `save` and `load` functions.

    Parameters
    ---
    name: str
        The name to use for the object in the registry.

    warn: bool
        Whether to warn if the object is already in the registry.

    Returns
    ---
    decorator: Callable
        A decorator that can be used to store the object in the registry.
    """

    def decorator(obj):
        key = f"__TYPE__:{name}"

        if key in _SAVEABLE_TYPES and warn:
            print(f"Warning: Overwriting existing saveable type {name}")

        _SAVEABLE_TYPES[key] = obj
        _SAVEABLE_TYPES_INV[obj] = key

        return obj

    return decorator


class NotSerializableError(TypeError):
    """
    Error raised when attempting to serialize an object that is not
    serializable.
    """


# indicate that the value is an array and is therefore saved separately
_ARRAY = "__ARRAY_PLACEHOLDER__"


def _lookup_to_str(lookup: ItemLookup | AttrLookup) -> dict:
    """
    Convert a lookup object to a dictionary that can be serialized.
    """
    key = lookup.key

    if not isinstance(key, (int, str)):
        if key not in _SAVEABLE_TYPES_INV:
            error = f"Type {key} is not serializable"
            raise NotSerializableError(error)

        key = _SAVEABLE_TYPES_INV[key]

    lookup_type = {ItemLookup: "item", AttrLookup: "attr"}[type(lookup)]
    return {"key": key, "type": lookup_type}


def _str_to_lookup(data) -> ItemLookup | AttrLookup:
    """
    Inverse of `_lookup_to_str`.
    """
    key = data["key"]

    if key in _SAVEABLE_TYPES:
        key = _SAVEABLE_TYPES[key]

    lookup_type = data["type"]
    return {"item": ItemLookup, "attr": AttrLookup}[lookup_type](key)


def save(module, save_path: str):
    """
    Save a module to a file. All leaves, types and lookups in the module must
    be serializable, i.e. jax or numpy array, int, float, bool, str, None, or
    nested versions of dicts, lists, tuples.
    Values and types that were added to the registry using the `saveable`
    decorator can also be serialized.
    """
    flat = flatten(module)

    keys = []
    arrs = []
    sims = []

    for key, val in flat.items():
        key = [_lookup_to_str(k) for k in key.path]
        keys.append(key)

        if isinstance(val, (jax.Array, onp.ndarray)):
            arrs.append(val)
            sims.append(_ARRAY)
            continue

        if not isinstance(val, (str, int, float, bool, type(None))):
            if val not in _SAVEABLE_TYPES_INV:
                error = f"Type {val} is not serializable"
                raise NotSerializableError(error)

            val = _SAVEABLE_TYPES_INV[val]

        sims.append(val)

    data = {"keys": keys, "sims": sims}
    data = json.dumps(data, indent=2)

    arr_dict = {"data": data}

    for i, arr in enumerate(arrs):
        arr_dict[str(i)] = arr

    jnp.savez(save_path, **arr_dict)


def load(load_path: str):
    """
    Load a object from a file that was saved using the `save` function.

    Module which were marked as serializable using the `saveable` decorator
    must be imported before calling this function, otherwise this function
    will not know how to deserialize them.

    Parameters
    ---
    load_path: str
        The path to the file to load.

    Returns
    ---
    Any
        The deserialized object.

    """
    file = dict(jnp.load(load_path))

    data = json.loads(str(file.pop("data")))

    sims = data["sims"]
    keys = data["keys"]

    arrs = []

    for i in range(len(file)):
        key = str(i)
        arr = file.get(key)

        arrs.append(arr)

    flat = {}
    i = 0

    for key, sim in zip(keys, sims, strict=True):
        path = PathLookup(tuple(_str_to_lookup(k) for k in key))

        if sim == _ARRAY:
            flat[path] = jnp.array(arrs[i])
            i += 1
            continue

        if sim in _SAVEABLE_TYPES:
            sim = _SAVEABLE_TYPES[sim]

        flat[path] = sim

    return unflatten(flat)


# make the regular python types serializable
saveable("list")(list)
saveable("dict")(dict)
saveable("tuple")(tuple)

# make the module types from ._module serializable, since we cannot make
# them serializable in the module itself, because this would create a
# circular dependency
saveable("flarejax.Module")(Module)
saveable("flarejax.MethodWrap")(MethodWrap)

# make common activation functions serializable
saveable("jax.nn.celu")(jnn.celu)
saveable("jax.nn.elu")(jnn.elu)
saveable("jax.nn.gelu")(jnn.gelu)
saveable("jax.nn.glu")(jnn.glu)
saveable("jax.nn.hard_sigmoid")(jnn.hard_sigmoid)
saveable("jax.nn.hard_silu")(jnn.hard_silu)
saveable("jax.nn.hard_swish")(jnn.hard_swish)
saveable("jax.nn.hard_tanh")(jnn.hard_tanh)
saveable("jax.nn.leaky_relu")(jnn.leaky_relu)
saveable("jax.nn.log_sigmoid")(jnn.log_sigmoid)
saveable("jax.nn.log_softmax")(jnn.log_softmax)
saveable("jax.nn.logsumexp")(jnn.logsumexp)
saveable("jax.nn.standardize")(jnn.standardize)
saveable("jax.nn.relu")(jnn.relu)
saveable("jax.nn.relu6")(jnn.relu6)
saveable("jax.nn.selu")(jnn.selu)
saveable("jax.nn.sigmoid")(jnn.sigmoid)
saveable("jax.nn.soft_sign")(jnn.soft_sign)
saveable("jax.nn.softmax")(jnn.softmax)
saveable("jax.nn.softplus")(jnn.softplus)
saveable("jax.nn.sparse_plus")(jnn.sparse_plus)
saveable("jax.nn.sparse_sigmoid")(jnn.sparse_sigmoid)
saveable("jax.nn.silu")(jnn.silu)
saveable("jax.nn.swish")(jnn.swish)
saveable("jax.nn.squareplus")(jnn.squareplus)
saveable("jax.nn.mish")(jnn.mish)

# make the jax data types serializable
saveable("jax.numpy.bool_")(jnp.bool_)
saveable("jax.numpy.complex64")(jnp.complex64)
saveable("jax.numpy.complex128")(jnp.complex128)
saveable("jax.numpy.float16")(jnp.float16)
saveable("jax.numpy.float32")(jnp.float32)
saveable("jax.numpy.float64")(jnp.float64)
saveable("jax.numpy.bfloat16")(jnp.bfloat16)
saveable("jax.numpy.int8")(jnp.int8)
saveable("jax.numpy.int16")(jnp.int16)
saveable("jax.numpy.int32")(jnp.int32)
saveable("jax.numpy.int64")(jnp.int64)
saveable("jax.numpy.uint8")(jnp.uint8)
saveable("jax.numpy.uint16")(jnp.uint16)
saveable("jax.numpy.uint32")(jnp.uint32)
saveable("jax.numpy.uint64")(jnp.uint64)
