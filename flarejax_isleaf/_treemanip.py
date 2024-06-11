import dataclasses
from typing import Hashable

# from ._typecheck import typecheck


@dataclasses.dataclass(frozen=True)
class ModuleAttr:
    name: str
    type: type
    tree: Hashable
    hint: Hashable | None = None


@dataclasses.dataclass(frozen=True)
class MappingKey:
    type: type
    tree: Hashable | None = None


@dataclasses.dataclass(frozen=True)
class MappingVal:
    name: str
    type: type
    tree: Hashable | None = None


@dataclasses.dataclass(frozen=True)
class TupleIndex:
    name: int
    type: type
    tree: Hashable | None = None


import jax.tree_util as jtu

jtu.tree_flatten_with_path
jtu.default_registry
