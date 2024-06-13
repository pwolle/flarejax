from types import MappingProxyType
from typing import Any

import jax.tree_util as jtu

# Global module registry to store all registered modules.
# The registry is a dictionary with module names as keys and the module classes
# as values. The module names are the names of the classes by default, but can
# be overridden by setting a '__module_name' attribute on the class.
MODULE_REGISTRY = {
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "NoneType": type(None),
    "MappingProxyType": MappingProxyType,
}


def _flatten_mappingproxy_with_keys(
    mapping: MappingProxyType,
) -> tuple[tuple[tuple[jtu.DictKey, MappingProxyType], ...], tuple[str]]:
    children = []
    aux_data = []

    for key, value in mapping.items():
        children.append((jtu.DictKey(key), value))
        aux_data.append(key)

    return tuple(children), tuple(aux_data)


def _unflatten_mappingproxy(_, children) -> MappingProxyType:
    return MappingProxyType(dict(children))


jtu.register_pytree_with_keys(
    MappingProxyType,
    _flatten_mappingproxy_with_keys,
    _unflatten_mappingproxy,
)


def _flatten_slice_with_keys(
    slice_: slice,
) -> tuple[tuple[tuple[jtu.GetAttrKey, Any], ...], None]:
    return (
        (
            (jtu.GetAttrKey("start"), slice_.start),
            (jtu.GetAttrKey("stop"), slice_.stop),
            (jtu.GetAttrKey("step"), slice_.step),
        ),
        None,
    )


def _unflatten_slice(_, children) -> slice:
    return slice(*children)


jtu.register_pytree_with_keys(
    slice,
    _flatten_slice_with_keys,
    _unflatten_slice,
)
