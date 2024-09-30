"""
Combining neural network modules.
"""

from typing import Callable, Self

import jax
import jax.numpy as jnp
import jax.random as jrn
from jaxtyping import PRNGKeyArray

from .._module import Module
from .._serial import saveable
from .._tcheck import typecheck

__all__ = [
    "Sequential",
    "Add",
    "Multiply",
    "Constant",
]


@saveable("flarejax.Sequential")
class Sequential(Module):
    """
    Call a sequence of modules in order.
    The first argument is updated, while the args and kwargs are passed to
    each module.

    Parameters
    ---
    modules: Callable[..., jax.Array]
        Modules to call in order.

    """

    modules: list[Callable]

    def __init__(self, *modules: Callable):
        self.modules = list(modules)

    def __call__(self, x, *args, **kwargs):
        for module in self.modules:
            x = module(x, *args, **kwargs)

        return x


@saveable("flarejax.Add")
class Add(Sequential):
    """
    Add the output of a sequence of modules to the input.
    The first argument is updated, while the args and kwargs are passed to
    each module.

    Parameters
    ---
    modules: Callable[..., jax.Array]
        Modules to call in order.

    """

    def __call__(self, x, *args, **kwargs):
        for module in self.modules:
            x = x + module(x, *args, **kwargs)

        return x


@saveable("flarejax.Multiply")
class Multiply(Sequential):
    """
    Multiply the output of a sequence of modules by the input.
    The first argument is updated, while the args and kwargs are passed to
    each module.

    Parameters
    ---
    modules: Callable[..., jax.Array]
        Modules to call in order.

    """

    def __call__(self, x, *args, **kwargs):
        for module in self.modules:
            x = x * module(x, *args, **kwargs)

        return x


@saveable("flarejax.Constant")
@typecheck
class Constant(Module):
    """
    Return a constant value. Ignores all inputs.

    Parameters
    ---
    value: jax.Array
        Constant value to return.

    """

    value: jax.Array

    def __init__(self, value):
        self.value = value

    @classmethod
    def random_normal(
        cls,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        std: float = 1.0,
    ) -> Self:
        """
        Initialize with random normal values.

        Parameters
        ---
        key: PRNGKeyArray
            The key to use for random number generation.

        shape: tuple[int, ...]
            The shape of the array.

        std: float
            The standard deviation of the normal distribution.

        Returns
        ---
        Constant
            The initialized module.
        """
        return cls(jrn.normal(key, shape) * std)

    @classmethod
    def random_uniform(
        cls,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        low: float = -1.0,
        high: float = 1.0,
    ) -> Self:
        """
        Initialize with random uniform values.

        Parameters
        ---
        key: PRNGKeyArray
            The key to use for random number generation.

        shape: tuple[int, ...]
            The shape of the array.

        low: float
            The lower bound of the uniform distribution.

        high: float
            The upper bound of the uniform distribution.

        Returns
        ---
        Constant
            The initialized module.
        """
        return cls(jrn.uniform(key, shape, minval=low, maxval=high))

    @classmethod
    def full(cls, x: float = 0.0, shape: tuple[int, ...] = ()) -> Self:
        """
        Initialize with a constant value.

        Parameters
        ---
        x: float
            The constant value.

        shape: tuple[int, ...]
            The shape of the array.

        Returns
        ---
        Constant
            The initialized module.
        """
        return cls(jnp.full(shape, x))

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return the shape of the constant value.
        """
        return self.value.shape

    @property
    def dtype(self) -> jnp.dtype:
        """
        Return the dtype of the constant value.
        """
        return self.value.dtype

    def __call__(self, *args, **kwargs) -> jax.Array:
        """
        Return the constant value, ignoring all inputs.
        """
        return self.value


@saveable("flarejax.Identity")
class Identity(Module):
    """
    Return the input. Ignores args and kwargs.
    Useful combining modules, e.g. creating a simple residual
    connection by combining a module with an Identity module using Add.

    Parameters
    ---
    takes no parameters
    """

    def __call__(self, x, *args, **kwargs):
        return x
