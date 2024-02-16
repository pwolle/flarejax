import abc

import jax.tree_util as jtree

__all__ = [
    "PyTreeBase",
]


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
        jtree.register_pytree_with_keys_class(cls)

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


class PyTreeBase(metaclass=PyTreeMeta):
    pass
