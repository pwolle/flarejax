import abc
import dataclasses
from typing import Any, Callable, Self, dataclass_transform, Hashable, Sequence

import jax
import functools
import jax.tree_util as jtu

from ._trees import MODULE_REGISTRY
from ._typecheck import typecheck
from ._utils import array_summary
from types import MappingProxyType

__all__ = [
    "field",
    "get_module_name",
    "Module",
    "BoundMethod",
    "BoundMethodWrap",
]


@functools.wraps(dataclasses.field)
def field(*, static: bool = False, **kwargs):
    metadata = kwargs.pop("metadata", {})
    metadata = dict(metadata)  # perform copy and make mutable
    metadata["static"] = static
    metadata = MappingProxyType(metadata)  # make immutable again
    return dataclasses.field(metadata=metadata, **kwargs)


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
@dataclass_transform(frozen_default=True)
class FrozenDataclassMeta(abc.ABCMeta, type):
    """
    This metaclass makes all its subclasses frozen dataclasses and registers
    them as JAX PyTrees. The fields of the dataclass are the pytree leaves.

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
        for key, value in attrs.items():
            if name in ["FrozenDataclassBase", "Module", "BoundMethod"]:
                break

            if key == "tree_flatten_with_keys" or key == "tree_unflatten":
                continue

            if not callable(value):
                continue

            attrs[key] = BoundMethodWrap(value)

        cls_new = super().__new__(cls, name, bases, attrs)
        cls_new = dataclasses.dataclass(frozen=True, repr=False)(cls_new)

        cls_new.__init__ = typecheck(cls_new.__init__)

        if not kwargs.get("register", True):
            return cls_new

        module_name = get_module_name(cls_new)

        if module_name in MODULE_REGISTRY and not kwargs.get("replace", False):
            error = (
                f"Module with name '{module_name}' is already registered. "
                "Consider using a different name for the module or adding a"
                "'__module_name' class attribute to the module definition. "
                "In case you want to replace the existing module, set the "
                "'replace' keyword argument to 'True'."
            )
            raise ValueError(error)

        MODULE_REGISTRY[module_name] = cls_new

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
class FrozenDataclassBase(metaclass=FrozenDataclassMeta, register=False):
    @abc.abstractmethod
    def tree_flatten_with_keys(self: Self):
        error = "Abstract method 'tree_flatten_with_keys' must be implemented."
        raise NotImplementedError(error)

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux_data, children: tuple[Any, ...]) -> Self:
        error = "Abstract method 'tree_unflatten' must be implemented."
        raise NotImplementedError(error)


@typecheck
class Module(FrozenDataclassBase):
    def tree_flatten_with_keys(
        self: Self,
    ) -> tuple[
        tuple[tuple[jtu.GetAttrKey, Any], ...],
        tuple[tuple[str, Hashable], ...],
    ]:
        children = []
        aux_data = []

        for field in sorted(dataclasses.fields(self), key=lambda f: f.name):
            if field.metadata.get("static", False):
                value = getattr(self, field.name)

                if not isinstance(value, Hashable):
                    error = f"Non-hashable static data '{field.name}: {value}'"
                    raise ValueError(error)

                aux_data.append((field.name, value))
                continue

            v = getattr(self, field.name)
            k = jtu.GetAttrKey(field.name)
            children.append((k, v))

        return tuple(children), tuple(aux_data)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Sequence[tuple[str, Hashable]],
        children: tuple[Any, ...],
    ) -> Self:
        aux_data_dict = dict(aux_data)
        kwargs = {}

        fields = sorted(dataclasses.fields(cls), key=lambda f: f.name)

        for child, field in zip(children, fields):
            if field.name in aux_data_dict:

                kwargs[field.name] = aux_data_dict[field.name]
                continue

            kwargs[field.name] = child

        return cls(**kwargs)

    def __repr__(self) -> str:
        head = f"{self.__class__.__name__}("

        if len(dataclasses.fields(self)) == 0:
            return head + ")"

        body = []

        for field in dataclasses.fields(self):
            value = getattr(self, field.name)

            if isinstance(value, jax.Array):
                value = array_summary(value)
            else:
                value = repr(value)

            body.append(f"{field.name}={value}")

        body = ",\n".join(body)
        body = "\n" + body

        body = body.replace("\n", "\n  ")
        body = body + "\n"

        tail = ")"
        return head + body + tail


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
