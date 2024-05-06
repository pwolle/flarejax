from beartype.claw import beartype_this_package

beartype_this_package()

from ._config import ConfigMapping, ConfigSequence
from ._frozen import FrozenMapping, FrozenMappingHashable, FrozenSequence
from ._module import FieldKey, Module, field
from ._mtypes import (
    Jit,
    ModuleMapping,
    ModuleSequence,
    Partial,
    Sequential,
    VMap,
)
from ._serial import load_module, save_module

__version__ = "0.2.3"


__all__ = [
    "ConfigMapping",
    "ConfigSequence",
    "FrozenMapping",
    "FrozenMappingHashable",
    "FrozenSequence",
    "Module",
    "FieldKey",
    "field",
    "Jit",
    "ModuleMapping",
    "ModuleSequence",
    "Partial",
    "Sequential",
    "VMap",
    "load_module",
    "save_module",
    "__version__",
]
