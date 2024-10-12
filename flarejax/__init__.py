"""
Base module for FlareJAX. Manipulation and serialization of Modules.
"""

from . import net, opt
from .flr._filter import filter_jit
from .flr._lookup import AttrLookup, ItemLookup, PathLookup
from .flr._module import Module, flatten, unflatten
from .flr._random import RandomKey
from .flr._serial import load, save, saveable
from .flr._tcheck import typecheck
from .flr._transform import Jit, Partial, Vmap

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
