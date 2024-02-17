import abc
from typing import Self
import jax.tree_util as jtree

__all__ = [
    "PyTreeWithKeys",
]


class PyTreeWithKeysMeta(abc.ABCMeta, type):
    """
    Metaclass for registering a class as Jax PyTree with keys.
    """

    def __init__(cls, name, bases, attrs, **kwargs) -> None:
        """
        Register the class as a PyTree node.
        """
        super().__init__(name, bases, attrs, **kwargs)
        jtree.register_pytree_with_keys_class(cls)


class PyTreeWithKeys(metaclass=PyTreeWithKeysMeta):
    @abc.abstractmethod
    def tree_flatten_with_keys(self):
        """
        Flatten the PyTree node into a tuple of children and aux data.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the PyTree node from a tuple of children and aux data.
        """
        raise NotImplementedError
