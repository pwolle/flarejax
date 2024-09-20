from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jrn
from jaxtyping import PRNGKeyArray

from ._module import Module

__all__ = [
    "Sequential",
    "Add",
    "Multiply",
    "Constant",
]


class Sequential(Module):
    modules: list[Callable[..., jax.Array]]

    def __init__(self, *modules):
        self.modules = list(modules)

    def __call__(self, x, *args, **kwargs):
        for module in self.modules:
            x = module(x, *args, **kwargs)

        return x

    def __repr__(self):
        head = f"{type(self).__name__}(["

        if not self.modules:
            return head + "])"

        body = []

        for module in self.modules:
            rep = repr(module)
            body.append(rep)

        body = ",\n".join(body)
        body = "\n" + body

        # add indentation
        body = body.replace("\n", "\n  ")
        body = body + ",\n"

        tail = "])"
        return head + body + tail


class Add(Module):
    module: Callable[..., jax.Array]

    def __init__(self, module):
        self.module = module

    def __call__(self, x, *args, **kwargs):
        return x + self.module(x, *args, **kwargs)


class Multiply(Module):
    module: Callable[..., jax.Array]

    def __init__(self, module):
        self.module = module

    def __call__(self, x, *args, **kwargs):
        return x * self.module(x, *args, **kwargs)


class Constant(Module):
    value: jax.Array

    def __init__(self, value):
        self.value = value

    @classmethod
    def random_normal(
        cls,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        std: float = 1.0,
    ):
        return cls(jrn.normal(key, shape) * std)

    @classmethod
    def random_uniform(
        cls,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        low: float = -1.0,
        high: float = 1.0,
    ):
        return cls(jrn.uniform(key, shape, minval=low, maxval=high))

    @classmethod
    def full(cls, x: float = 0.0, shape: tuple[int, ...] = ()):
        return cls(jnp.full(shape, x))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.value.dtype

    def __call__(self, *_):
        return self.value
