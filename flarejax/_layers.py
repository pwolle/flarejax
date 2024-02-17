from typing import Callable, Self, Sequence, TypeAlias

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from typeguard import typechecked

from ._module import Module, ModuleList
from ._param import Param

Init: TypeAlias = Callable[[PRNGKeyArray, Sequence[int]], Array]


class Sequential(ModuleList):
    modules: list[Callable]

    @typechecked
    def __call__(self: Self, x):
        for module in self.modules:
            x = module(x)

        return x


@typechecked
def init_he(key: PRNGKeyArray, dims: Sequence[int]) -> Array:
    fan_in = np.prod(dims[:-1])
    stddev = np.sqrt(2 / fan_in)
    return jrandom.normal(key, dims, dtype=jnp.float32) * stddev


@typechecked
def init_zeros(_: PRNGKeyArray, dims: Sequence[int]) -> Array:
    return jnp.zeros(dims, dtype=jnp.float32)


@typechecked
class Linear(Module):
    w: Param[Array]
    b: Param[Array | None]

    def __init__(
        self: Self,
        key: PRNGKeyArray,
        dim_in: int,
        dim: int,
        *,
        bias: bool = True,
        init_w: Init = init_he,
        init_b: Init = init_zeros,
    ) -> None:
        key_w, key_b = jrandom.split(key)
        self.w = Param(init_w(key_w, (dim_in, dim)))
        self.b = Param(init_b(key_b, (dim,)) if bias else None)

    def __call__(self: Self, x: Float[Array, "*b _"]) -> Float[Array, "*b _"]:
        y = x @ self.w.value

        if self.b.value is not None:
            y += self.b.value

        return y


@typechecked
def layer_norm(x: Float[Array, "*b d"], eps: float = 1e-4) -> Float[Array, "*b d"]:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    var = jnp.maximum(0, var)
    inv = lax.rsqrt(var + eps)
    return (x - mean) * inv


@typechecked
class LayerNorm(Module):
    w: Param[Array]
    b: Param[Array]

    def __init__(self: Self, dim: int, *, eps: float = 1e-4) -> None:
        self.w = Param(jnp.ones((dim,)))
        self.b = Param(jnp.zeros((dim,)))
        self.eps = eps

    def __call__(self: Self, x: Float[Array, "*b d"]) -> Float[Array, "*b d"]:
        return layer_norm(x, self.eps) * self.w.value + self.b.value
