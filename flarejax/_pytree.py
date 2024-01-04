import abc
from typing import Any, Self

import jax.tree_util as jtree
import typeguard

__all__ = [
    "PyTreeBase",
]


@typeguard.typechecked
class PyTreeMeta(type, metaclass=abc.ABCMeta):
    """
    Metaclass for registering PyTree nodes. Automatically registers
    all subclasses as PyTrees.
    """

    def __init__(cls, name, bases, attrs, **kwargs) -> None:
        """
        Register the class as a PyTree node.
        """
        super().__init__(name, bases, attrs, **kwargs)
        jtree.register_pytree_node_class(cls)

    @abc.abstractmethod
    def tree_flatten(
        self: Self,
    ) -> tuple[tuple[Any, ...], Any]:
        """
        Flatten the PyTree node into a tuple of children and aux data.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(
        cls: type[Self],
        aux_data: Any,
        children: tuple[Any, ...],
    ) -> Self:
        """
        Unflatten the PyTree node from a tuple of children and aux data.
        """
        raise NotImplementedError


class PyTreeBase(metaclass=PyTreeMeta):
    pass
