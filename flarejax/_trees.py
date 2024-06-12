import jax.tree_util as jtu

from types import MappingProxyType


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


def _unflatten_mappingproxy(children, _):
    return MappingProxyType(dict(children))


jtu.register_pytree_with_keys(
    MappingProxyType,
    _flatten_mappingproxy_with_keys,
    _unflatten_mappingproxy,
)
