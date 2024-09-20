import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float

from ._module import Module


class LayerNorm(Module):
    weight: Float[Array, "dim"] | None
    bias: Float[Array, "dim"] | None

    def __init__(self, eps: float = 1e-4, axis: int = -1) -> None:
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

        m = x.mean(axis=self.axis, keepdims=True)
        x = x - m

        v = x.var(axis=self.axis, keepdims=True)
        v = jnp.maximum(v, self.eps)

        r = lax.rsqrt(v)
        x = x * r

        x = x * self.weight + self.bias
        return x


class RMSNorm(Module):
    weight: Float[Array, "dim"] | None
    bias: Float[Array, "dim"] | None

    def __init__(self, eps: float = 1e-4, axis: int = -1) -> None:
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