import abc
import dataclasses
import functools
import inspect
from typing import (
    Any,
    Callable,
    Hashable,
    Mapping,
    Self,
    dataclass_transform,
)

import jax
import jax.tree_util as jtu

from ._config import ConfigMapping, ConfigSequence, coerce_config_type
from ._frozen import FrozenMappingHashable
from ._modify import AtWrapper
from ._typecheck import typecheck

__all__ = [
    "GLOBAL_MODULE_REGISTRY",
    "get_module_name",
    "field",
    "FrozenDataclassMeta",
    "FieldKey",
    "Module",
    "AtWrapper",
    "is_leaf",
]

# Global module registry to store all registered modules.
# The registry is a dictionary with module names as keys and the module classes
# as values. The module names are the names of the classes by default, but can
# be overridden by setting a '__module_name' attribute on the class.
GLOBAL_MODULE_REGISTRY = {
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "NoneType": type(None),
}


@typecheck
def get_module_name(cls: type) -> str:
    """
    Get the name of a given module class. This is the class name by default,
    but can be overridden by setting a '__module_name' attribute on the class.

    The double underscore invokes python's name mangling which makes sure,
    that the same name is not used for two modules, just because one inherits
    from the other.

    Parameters:
    ---
    cls: type
        The class to get the module name for.

    Returns:
    ---
    str
        The name of the module class.
    """
    # attribute name of '__module_name' after private name mangling
    mangled_name = f"_{cls.__name__}__module_name"

    if hasattr(cls, mangled_name):
        name = getattr(cls, mangled_name)

        if not isinstance(name, str):
            error = f"Expected __module_name to be a string, but got: {name}"
            raise TypeError(error)

        return name

    return cls.__name__


@typecheck
def field(
    *,
    grad: bool = True,
    meta: Mapping | None = None,
    **field_kwargs,
) -> Any:
    meta = {} if meta is None else dict(meta)

    for v in meta.values():
        if not isinstance(v, Hashable):
            error = f"Metadata values must be hashable, but got: {v}"
            raise TypeError(error)

    assert "grad" not in meta, "'grad' is a reserved metadata key"
    meta["grad"] = grad

    return dataclasses.field(
        metadata=FrozenMappingHashable(meta),  # type: ignore
        **field_kwargs,
    )


@typecheck
@dataclass_transform(frozen_default=True)
class FrozenDataclassMeta(abc.ABCMeta, type):
    """
    This metaclass makes all its subclasses frozen dataclasses and registers
    them as Jax PyTrees.

    Parameters:
    ---
    register: bool
        Whether to register the class in the global module registry.
    """

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, Any],
        **kwargs: Any,
    ):
        """
        Convert the given class into a frozen dataclass.
        """
        # Make bound methods modules. however, exclude the base classes, as they
        # are defined before the BoundMethod Module is defined.
        if name not in {
            "FrozenDataclassBase",
            "Module",
            "BoundMethod",
            "AtWrapper",
        }:
            for key, value in attrs.items():
                if key == "tree_flatten_with_keys" or key == "tree_unflatten":
                    continue

                if key.startswith("__"):
                    continue

                if not callable(value):
                    continue

                if isinstance(value, Module):
                    continue

                attrs[key] = BoundMethodWrap(value)

        cls_new = super().__new__(cls, name, bases, attrs)
        cls_new = dataclasses.dataclass(frozen=True)(cls_new)

        if kwargs.get("register", True):
            module_name = get_module_name(cls_new)

            if module_name in GLOBAL_MODULE_REGISTRY:
                error = (
                    f"Module with name '{module_name}' is already registered."
                    "Consider using a different name for the module or adding a"
                    "'__module_name' class attribute to the module definition."
                )
                raise ValueError(error)

            GLOBAL_MODULE_REGISTRY[module_name] = cls_new

        return cls_new

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, Any],
        **_: Any,
    ) -> None:
        super().__init__(name, bases, attrs)
        jtu.register_pytree_with_keys_class(cls)


@typecheck
@dataclasses.dataclass(frozen=True)
class FieldKey:
    """
    Dataclass to store metadata about a field in the key of the PyTree structure.

    Attributes:
    ---
    TODO
    """

    name: str  # name of the field
    type: type  # type of parent module
    tree: ConfigMapping | ConfigSequence = ConfigMapping()  # static part
    hint: FrozenMappingHashable = FrozenMappingHashable()  # metadata

    # cache the hash value since it is O(n) to compute and it will not change,
    # because the object is immutable
    @functools.cached_property
    def _hash(self) -> int:
        return hash((self.name, self.tree, self.hint))

    def __hash__(self) -> int:
        return self._hash


@typecheck
class FrozenDataclassBase(metaclass=FrozenDataclassMeta, register=False):
    """
    Abstract base class for creating frozen dataclasses with support for the Jax
    PyTree-with-keys API.

    The class must implement the 'tree_flatten_with_keys' and 'tree_unflatten'
    methods to support the Jax PyTree-with-keys API.
    """

    @abc.abstractmethod
    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[tuple[tuple[FieldKey, Any], ...], ConfigMapping]:
        """
        Flatten the PyTree node into a tuple of children and aux data.
        """
        error = "Abstract method 'tree_flatten_with_keys' must be implemented."
        raise NotImplementedError(error)

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(
        cls,
        aux_data: ConfigMapping | ConfigSequence,
        children: tuple[Any, ...],
    ) -> Self:
        """
        Unflatten the PyTree node from a tuple of children and aux data.
        """
        error = "Abstract method 'tree_unflatten' must be implemented."
        raise NotImplementedError(error)


@typecheck
@functools.cache
def sorted_fields(cls: type) -> list[dataclasses.Field]:
    if not inspect.isclass(cls):
        error = f"Expected a class, but got: {cls}"
        raise TypeError(error)

    if not dataclasses.is_dataclass(cls):
        error = f"Expected a dataclass, but got: {cls}"
        raise TypeError(error)

    fields = dataclasses.fields(cls)
    fields = sorted(fields, key=lambda f: f.name)
    return fields


@typecheck
def is_leaf(value: Any) -> bool:
    return isinstance(value, (Module, jax.Array))


@typecheck
class Module(FrozenDataclassBase, register=False):
    def __post_init__(self: Self) -> None:
        for f in dataclasses.fields(type(self)):
            value = getattr(self, f.name)

            if is_leaf(value):
                continue

            value = coerce_config_type(value)
            object.__setattr__(self, f.name, value)

    def __frozen_set_attr__(self: Self, k: str, v: Any, /) -> Self:
        return dataclasses.replace(self, **{k: v})

    def __frozen_set_item__(self: Self, k: Any, v: Any, /) -> Any:
        error = f"Setting items is not supported for '{type(self)}'"
        raise NotImplementedError(error)

    def __frozen_del_item__(self: Self, k: Any, /) -> Any:
        error = f"Deleting items is not supported for '{type(self)}'"
        raise NotImplementedError(error)

    @property
    def at(self: Self) -> "AtWrapper":
        return AtWrapper(_at_module=self)

    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[tuple[tuple[FieldKey, Any], ...], ConfigMapping]:
        """
        Flatten the PyTree node into a tuple of the leaves with field keys and
        a tuple with all of the non-leaf fields.
        """
        fields_static = []
        fields_active = []

        for field_ in sorted_fields(type(self)):
            value = getattr(self, field_.name)

            if is_leaf(value):
                fields_active.append(field_)
            else:
                fields_static.append(field_)

        static = {}
        for field_ in fields_static:
            static[field_.name] = getattr(self, field_.name)

        static = ConfigMapping(static)

        active = []
        for field_ in fields_active:
            value = getattr(self, field_.name)

            key = FieldKey(
                name=field_.name,
                type=type(self),
                tree=static,
                hint=FrozenMappingHashable(field_.metadata),
            )
            active.append((key, value))

        return tuple(active), static

    @classmethod
    def tree_unflatten(
        cls,
        static: ConfigMapping,
        active: tuple[Any, ...],
    ) -> Self:
        """
        Unflatten the PyTree node from a tuple of children and aux data.
        """
        kwargs = {}
        i = 0

        for field_ in sorted_fields(cls):
            if field_.name in static:
                kwargs[field_.name] = static[field_.name]

            else:
                kwargs[field_.name] = active[i]
                i += 1

        return cls(**kwargs)


@typecheck
class BoundMethod(Module):
    module: Module
    method: str

    @classmethod
    def init(cls, module_method) -> Self:
        if not hasattr(module_method, "__self__"):
            error = f"Method {module_method} is not bound to a module."
            raise ValueError(error)

        module = module_method.__self__

        if not callable(module_method):
            error = f"Method {module_method} is not callable."
            raise ValueError(error)

        if not isinstance(module, Module):
            error = f"Object {module} is not a Module."
            raise ValueError(error)

        return cls(
            module=module,
            method=module_method.__name__,
        )

    # maintain compatibility with the original python bound method type
    @property
    def __self__(self: Self) -> Module:
        return self.module

    def __call__(self: Self, *args, **kwargs):
        method = getattr(type(self.module), self.method)
        return method(self.module, *args, **kwargs)


@typecheck
@dataclasses.dataclass(frozen=True)
class BoundMethodWrap:
    method: Callable

    def __get__(self: Self, instance, owner=None) -> Callable | BoundMethod:
        if instance is None:
            return self.method

        method = self.method.__get__(instance, owner)
        return BoundMethod.init(method)
