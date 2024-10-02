"""
Base module for FlareJAX. Manipulation and serialization of Modules.
"""

from . import net, opt
from ._filter import filter_jit
from ._lookup import AttrLookup, ItemLookup, PathLookup
from ._module import Module, flatten, unflatten
from ._random import RandomKey
from ._serial import load, save, saveable
from ._tcheck import typecheck
from ._transform import Jit, Partial, Vmap

__all__ = [
    "net",
    "opt",
    "filter_jit",
    "ItemLookup",
    "AttrLookup",
    "PathLookup",
    "Module",
    "flatten",
    "unflatten",
    "RandomKey",
    "load",
    "save",
    "saveable",
    "typecheck",
    "Jit",
    "Partial",
    "Vmap",
]
