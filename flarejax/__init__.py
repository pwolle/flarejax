from ._config import ConfigMapping, ConfigSequence
from ._frozen import FrozenMappingHashable, FrozenSequence
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

__version__ = "0.2.0"


__all__ = [
    "ConfigMapping",
    "ConfigSequence",
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


from beartype.claw import beartype_this_package

beartype_this_package()
