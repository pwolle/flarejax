"""
Normalization layers.
"""

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .._module import Module
from .._serial import saveable


__all__ = [
    "LayerNorm",
    "RMSNorm",
]


@saveable("flarejax.LayerNorm")
class LayerNorm(Module):
    weight: Float[Array, "dim"] | None
    bias: Float[Array, "dim"] | None

    def __init__(
        self,
        eps: float = 1e-6,
        axis: int = -1,
        offset: bool = True,
    ) -> None:
        self.eps = eps
        self.axis = axis
        self.offset = offset

        self.weight = None
        self.bias = None

    def _build(self, x) -> None:
        self.weight = jnp.zeros(x.shape[-1], dtype=x.dtype)
        self.bias = jnp.zeros(x.shape[-1], dtype=x.dtype)

    def __call__(
        self,
        x: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        if self.weight is None or self.bias is None:
            self._build(x)

        assert self.weight is not None
        assert self.bias is not None

        m = x.mean(axis=self.axis, keepdims=True)
        x = x - m

        v = x.var(axis=self.axis, keepdims=True)
        v = jnp.maximum(v, self.eps)

        r = lax.rsqrt(v)
        x = x * r

        weight = self.weight + 1 if self.offset else self.weight
        x = x * weight + self.bias
        return x


@saveable("flarejax.RMSNorm")
class RMSNorm(Module):
    weight: Float[Array, "dim"] | None
    bias: Float[Array, "dim"] | None

    def __init__(self, eps: float = 1e-6, axis: int = -1) -> None:
        self.eps = eps
        self.axis = axis

        self.weight = None
        self.bias = None

    def _build(self, x) -> None:
        self.weight = jnp.ones(x.shape[-1], dtype=x.dtype)
        self.bias = jnp.zeros(x.shape[-1], dtype=x.dtype)

    def __call__(
        self,
        x: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        if self.weight is None or self.bias is None:
            self._build(x)

        v = (x * x).mean(axis=self.axis, keepdims=True)
        v = jnp.maximum(v, self.eps)

        r = lax.rsqrt(v)
        x = x * r

        x = x * self.weight + self.bias
        return x
