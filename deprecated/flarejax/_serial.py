import json
from types import MappingProxyType
from typing import Any, Mapping, Self

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as onp

from ._module import Module, get_module_name
from ._trees import MODULE_REGISTRY
from ._typecheck import typecheck


__all__ = [
    "save",
    "load",
]


@typecheck
def _treedef_to_dict(treedef, /) -> dict[str, Any] | None:
    """
    Convert a PyTree tree definition into a nested dictionary.
    """
    if treedef.node_data() is None:
        return None

    node_data = treedef.node_data()
    name = get_module_name(node_data[0])

    if name not in MODULE_REGISTRY:
        error = f"Module '{node_data[0]}' is not registered."
        raise ValueError(error)

    return {
        "__module_name": name,
        "aux_data": node_data[1],
        "children": [_treedef_to_dict(child) for child in treedef.children()],
    }


@typecheck
def _dict_to_treedef(data: Mapping | None, /) -> jtu.PyTreeDef:
    """
    Convert a nested dictionary created by 'treedef_to_dict' back into a PyTree.
    """
    if data is None:
        return jtu.PyTreeDef.make_from_node_data_and_children(
            jtu.default_registry, None, []
        )

    if data["__module_name"] not in MODULE_REGISTRY:
        error = f"Module with name '{data['__module_name']}' is not registered."
        raise ValueError(error)

    cls = MODULE_REGISTRY[data["__module_name"]]

    return jtu.PyTreeDef.make_from_node_data_and_children(
        jtu.default_registry,
        (cls, data["aux_data"]),
        (_dict_to_treedef(child) for child in data["children"]),
    )


@typecheck
def _ismapping(obj: Any) -> bool:
    try:
        _ = (lambda **kwargs: kwargs)(**obj)  # type: ignore
        return True
    except TypeError:
        return False


@typecheck
def _issequence(obj: Any) -> bool:
    try:
        _ = (lambda *args: args)(*obj)  # type: ignore
        return True
    except TypeError:
        return False


@typecheck
class ConfigEncoder(json.JSONEncoder):
    def default(self: Self, obj) -> dict[str, Any] | list[Any] | Any:
        if _ismapping(obj):
            return dict(obj)

        if _issequence(obj):
            return list(obj)

        return super().default(obj)


@typecheck
def save(path: str, tree: Module | list | dict | tuple | None) -> None:
    flat, treedef = jtu.tree_flatten(tree)
    arrays = {}
    primis = {}

    for i, value in enumerate(flat):
        if isinstance(value, (jax.Array, onp.ndarray)):
            arrays[str(i)] = value
            continue

        if isinstance(value, (bool, int, float, str)):
            primis[str(i)] = value
            continue

        error = f"Invalid value type: {value}"
        raise TypeError(error)

    treedef = _treedef_to_dict(treedef)
    jsondat = {"treedef": treedef, "primis": primis}

    jsonstr = json.dumps(jsondat, indent=2, cls=ConfigEncoder)
    arrays["json"] = jsonstr

    jnp.savez(path, **arrays)


@typecheck
class ConfigDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self: Self, obj) -> Any:
        if isinstance(obj, dict):
            return MappingProxyType(obj)

        if isinstance(obj, list):
            return tuple(obj)

        return obj


@typecheck
def load(path: str) -> Module | list | dict | tuple | None:
    arrays = dict(jnp.load(path))
    jsondat = json.loads(str(arrays.pop("json")), cls=ConfigDecoder)

    arrays = arrays | jsondat["primis"]
    flat = [jnp.array(arrays[k]) for k in sorted(arrays, key=int)]

    treedef = _dict_to_treedef(jsondat["treedef"])
    return jtu.tree_unflatten(treedef, flat)
