# %%
import dataclasses
from typing import Any, Protocol, runtime_checkable

import jax.tree_util as jtu


@runtime_checkable
class PathFilter(Protocol):
    def __call__(self, key, parent) -> "tuple[PathFilter, bool]": ...


@dataclasses.dataclass
class N:
    v: Any
    k: tuple
    s: tuple["N", ...]


def flatten_one_level_with_path(tree):
    return jtu.tree_flatten_with_path(
        tree,
        is_leaf=lambda x: x is not tree,
    )


_leaf_treedef = jtu.tree_structure(0)


def _is_leaf(x):
    return jtu.tree_structure(x) == _leaf_treedef


def tree_with_keys(tree) -> tuple[N, ...]:
    subnodes, _ = flatten_one_level_with_path(tree)
    subnodes_list = []

    for k, v in subnodes:
        subnodes_list.append(
            N(
                v,
                k,
                () if _is_leaf(v) else tree_with_keys(v),
            )
        )

    return tuple(subnodes_list)


def tree_with_path(x):
    tree = N(x, (), tree_with_keys(x))
    return tree


def tree_filter(tree: N, f: PathFilter):
    pass


nested = {
    "a": 1,
    "b": {
        "c": 2,
        "d": 3,
    },
    "e": [
        4,
        5,
        6,
    ],
    "f": (),
}


def filter_list(key, parent):
    return filter_list, isinstance(parent, list)


# class FilterList:
#     def __call__(self, x) -> tuple[Self, bool]:
#         return self, isinstance(x, list)

from pprint import pprint

pprint(tree_with_path(nested))

# %%
