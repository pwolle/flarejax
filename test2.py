# %%
import jax.tree_util as jtu

from typing import Protocol, runtime_checkable


@runtime_checkable
class PathFilter(Protocol):
    def __call__(self, key) -> "tuple[PathFilter, bool]": ...


def filter_flatten():
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
}

# nested["nested"] = nested


def isleaf(x):
    r = x is not nested
    print(f"testing {r}", x)
    return not r


# flattened, _ = jtu.tree_flatten_with_path(nested, is_leaf=lambda x: x is not nested)
flattened, _ = jtu.tree_flatten_with_path(nested)

for key_path, value in flattened:
    print(f"Value of tree{jtu.keystr(key_path)}: {value}")

# %%
