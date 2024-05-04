import abc
import dataclasses
import functools
import inspect
from typing import (
    Any,
    Callable,
    Hashable,
    Literal,
    Mapping,
    Self,
    dataclass_transform,
)

import jax
import jax.tree_util as jtu

from ._config import ConfigMapping
from ._frozen import FrozenMappingHashable

__all__ = [
    "GLOBAL_MODULE_REGISTRY",
    "get_module_name",
    "field",
    "FrozenDataclassMeta",
    "FieldKey",
    "Module",
    "AtWrapper",
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


@dataclass_transform(
    frozen_default=True,
    kw_only_default=True,
    order_default=False,
)
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
        cls_new = dataclasses.dataclass(
            frozen=True,
            kw_only=True,
            order=False,
        )(cls_new)

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


@dataclasses.dataclass(frozen=True)
class FieldKey:
    """
    Dataclass to store metadata about a field in the key of the PyTree structure.

    Attributes:
    ---
    name: str
        The name of the field.

    hint: type
        The type hint of the field.

    grad: bool
        Whether the field should be included when differentiating with respect
        to the dataclass instance.

    meta: FrozenMappingHashable[Hashable, Hashable]
        Additional metadata about the field.

    conf: FrozenMappingHashable[str | int, str | int | bool]
        Configuration options of the parent Module
    """

    name: str
    hint: type
    grad: bool
    attr: bool = True
    meta: FrozenMappingHashable = dataclasses.field(
        repr=False,
        default_factory=lambda: FrozenMappingHashable({}),
    )
    conf: FrozenMappingHashable = dataclasses.field(
        repr=False,
        default_factory=lambda: FrozenMappingHashable({}),
    )

    # cache the hash value since it is O(n) to compute and it will not change,
    # because the object is immutable
    @functools.cached_property
    def _hash(self) -> int:
        meta_unfolded = []
        for key in sorted(self.meta, key=hash):
            meta_unfolded.append((key, self.meta[key]))

        conf_unfolded = []
        for key in sorted(self.conf, key=hash):
            conf_unfolded.append((key, self.conf[key]))

        return hash(
            (
                self.name,
                self.hint,
                self.grad,
                tuple(meta_unfolded),
                tuple(conf_unfolded),
            )
        )

    def __hash__(self) -> int:
        return self._hash


class FrozenDataclassBase(metaclass=FrozenDataclassMeta, register=False):
    """
    Abstract base class for creating frozen dataclasses with support for the Jax
    PyTree-with-keys API.

    The class must implement the 'tree_flatten_with_keys' and 'tree_unflatten'
    methods to support the Jax PyTree-with-keys API.
    """

    @abc.abstractmethod
    def tree_flatten_with_keys(self: Self):
        """
        Flatten the PyTree node into a tuple of children and aux data.
        """
        error = "Abstract method 'tree_flatten_with_keys' must be implemented."
        raise NotImplementedError(error)

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        """
        Unflatten the PyTree node from a tuple of children and aux data.
        """
        error = "Abstract method 'tree_unflatten' must be implemented."
        raise NotImplementedError(error)


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


class Module(FrozenDataclassBase, register=False):
    config: ConfigMapping = field(
        repr=False,
        default_factory=lambda: ConfigMapping({"train": True}),
    )

    def __frozen_set_attr__(self: Self, adress: str, value: Any) -> Self:
        return dataclasses.replace(self, **{adress: value})

    def __frozen_set_item__(self: Self, adress: Any, value: Any) -> Any:
        error = f"Setting items is not supported for '{type(self)}'"
        raise NotImplementedError(error)

    def __frozen_del_item__(self: Self, adress: Any) -> Any:
        error = f"Deleting items is not supported for '{type(self)}'"
        raise NotImplementedError(error)

    @property
    def at(self: Self) -> "AtWrapper":
        return AtWrapper(_at_module=self)

    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[tuple[tuple[FieldKey, Any], ...], tuple[Hashable, ...]]:
        """
        Flatten the PyTree node into a tuple of the leaves with field keys and
        a tuple with all of the non-leaf fields.
        """
        active = []
        static = []

        for field_ in sorted_fields(type(self)):
            value = getattr(self, field_.name)

            if isinstance(value, (Module, jax.Array)):
                meta = FrozenMappingHashable(field_.metadata)
                meta, grad = meta.pop("grad")

                key = FieldKey(
                    name=field_.name,
                    hint=field_.type,
                    grad=grad,
                    attr=True,
                    meta=meta,
                    conf=self.config,
                )
                active.append((key, value))

            else:
                static.append(value)

        return tuple(active), tuple(static)

    @classmethod
    def tree_unflatten(
        cls, static: tuple[Hashable, ...], active: tuple[Any, ...]
    ) -> Self:
        """
        Unflatten the PyTree node from a tuple of children and aux data.
        """
        kwargs = {}
        active_index = 0
        static_index = 0

        for field_ in sorted_fields(cls):
            if field_.metadata.get("leaf", True):
                kwargs[field_.name] = active[active_index]
                active_index += 1
            else:
                kwargs[field_.name] = static[static_index]
                static_index += 1

        return cls(**kwargs)


EMPTY = object()


class AtWrapper(Module):
    _at_module: Module
    _at_adress: tuple[Any | str, ...] = field(default=())
    _at_lookup: tuple[Literal["attr", "item"], ...] = field(default=())

    def __post_init__(self: Self) -> None:
        if len(self._at_adress) != len(self._at_lookup):
            error = "Adress and lookup must have the same length."
            raise ValueError(error)

        if not all(lookup in ("attr", "item") for lookup in self._at_lookup):
            error = "Lookup must be either 'attr' or 'item'."
            raise ValueError(error)

    def __getattr__(self: Self, k: str) -> Self:
        return dataclasses.replace(
            self,
            _at_adress=self._at_adress + (k,),
            _at_lookup=self._at_lookup + ("attr",),
        )

    def __getitem__(self: Self, k: Any) -> Self:
        return dataclasses.replace(
            self,
            _at_adress=self._at_adress + (k,),
            _at_lookup=self._at_lookup + ("item",),
        )

    def set(self: Self, updated: Any) -> Module:
        values = [self._at_module]

        for adress, lookup in zip(
            self._at_adress[:-1],
            self._at_lookup[:-1],
            strict=True,
        ):
            if lookup == "attr":
                value = getattr(values[-1], adress)
                values.append(value)
                continue

            if lookup == "item":
                value = values[-1][adress]  # type: ignore
                values.append(value)
                continue

            error = f"Invalid lookup '{lookup}'."
            raise ValueError(error)

        for adress, value, lookup in zip(
            reversed(self._at_adress),
            reversed(values),
            reversed(self._at_lookup),
            strict=True,
        ):
            if lookup == "attr":
                updated = value.__frozen_set_attr__(adress, updated)
                continue

            if lookup == "item":
                if updated is EMPTY:
                    updated = value.__frozen_del_item__(adress)
                    continue

                updated = value.__frozen_set_item__(adress, updated)
                continue

            error = f"Invalid lookup '{lookup}'."
            raise ValueError(error)

        return updated


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


@dataclasses.dataclass(frozen=True)
class BoundMethodWrap:
    method: Callable

    def __get__(self: Self, instance, owner=None) -> Callable | BoundMethod:
        if instance is None:
            return self.method

        method = self.method.__get__(instance, owner)
        return BoundMethod.init(method)
