import json
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu

from ._config import ConfigEncoder, ConfigDecoder
from ._module import GLOBAL_MODULE_REGISTRY, Module, get_module_name
from ._typecheck import typecheck

__all__ = [
    "save_module",
    "load_module",
]


@typecheck
def treedef_to_dict(treedef, /) -> dict[str, Any] | None:
    """
    Convert a PyTree tree definition into a nested dictionary.
    """
    if treedef.node_data() is None:
        return None

    node_data = treedef.node_data()
    name = get_module_name(node_data[0])

    if name not in GLOBAL_MODULE_REGISTRY:
        error = f"Module with name '{name}' is not registered."
        raise ValueError(error)

    return {
        "__module_name": name,
        "aux_data": node_data[1],
        "children": [treedef_to_dict(child) for child in treedef.children()],
    }


@typecheck
def dict_to_treedef(data, /) -> jtu.PyTreeDef:
    """
    Convert a nested dictionary created by 'treedef_to_dict' back into a PyTree.
    """
    if data is None:
        return jtu.PyTreeDef.make_from_node_data_and_children(
            jtu.default_registry, None, []
        )

    if data["__module_name"] not in GLOBAL_MODULE_REGISTRY:
        error = f"Module with name '{data['__module_name']}' is not registered."
        raise ValueError(error)

    cls = GLOBAL_MODULE_REGISTRY[data["__module_name"]]

    return jtu.PyTreeDef.make_from_node_data_and_children(
        jtu.default_registry,
        (cls, data["aux_data"]),
        (dict_to_treedef(child) for child in data["children"]),
    )


@typecheck
def save_module(path: str, tree: Module | list | dict | tuple | None) -> None:
    """
    Save a module to disk.
    This is done by fist flattening the module using the PyTree API, then the
    leaves are saved to disk as numpy arrays and the tree structure is
    serialized to JSON and also saved in the same zip archive.
    """
    flat, treedef = jtu.tree_flatten(tree)
    arrays = {}

    for i, value in enumerate(flat):
        arrays[str(i)] = value

    treedef = treedef_to_dict(treedef)
    treedef = json.dumps(treedef, indent=2, cls=ConfigEncoder)

    arrays["treedef"] = treedef
    jnp.savez(path, **arrays)


@typecheck
def load_module(path: str) -> Module | list | dict | tuple | None:
    """
    Load a module from disk, which was saved using the 'save_module' function.

    Example
    ---
    >>> module = {"key": "value"}

    >>> from tempfile import NamedTemporaryFile
    >>>
    >>> with NamedTemporaryFile(suffix=".npz") as file:
    ...     save_module(file.name, module)
    ...     loaded = load_module(file.name)
    ...
    >>> assert loaded == module
    """
    arrays = dict(jnp.load(path))

    treedef = json.loads(str(arrays.pop("treedef")), cls=ConfigDecoder)
    treedef = dict_to_treedef(treedef)

    flat = [arrays[k] for k in sorted(arrays)]
    return jtu.tree_unflatten(treedef, flat)
