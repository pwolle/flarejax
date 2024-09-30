"""
Normalization layers.
"""

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped

from .._module import Module
from .._serial import saveable
from .._tcheck import typecheck


__all__ = [
    "LayerNorm",
    "RMSNorm",
]


@saveable("flarejax.LayerNorm")
class LayerNorm(Module):
    """
    Normalize the input along the last axis. Then add a learnable offset and
    scale by a learnable weight along the last axis.

    Parameters
    ---
    eps: float
        Epsilon value for numerical stability.

    offset: bool
        Whether to add 1 to the scale weight before multiplying. If true, the
        model is initialized to the identity function. If false, the model is
        initialized to the constant zero function.
    """

    @typecheck
    def __init__(
        self,
        eps: float = 1e-6,
        offset: bool = True,
    ) -> None:
        self.eps = eps
        self.offset = offset

        self.weight = None
        self.bias = None

    def _build(self, x) -> None:
        self.weight = jnp.zeros(x.shape[-1], dtype=x.dtype)
        self.bias = jnp.zeros(x.shape[-1], dtype=x.dtype)

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        if self.weight is None or self.bias is None:
            self._build(x)

        assert self.weight is not None
        assert self.bias is not None

        m = x.mean(axis=-1, keepdims=True)
        x = x - m

        v = x.var(axis=-1, keepdims=True)
        v = jnp.maximum(v, self.eps)

        r = lax.rsqrt(v)
        x = x * r

        weight = self.weight + 1 if self.offset else self.weight
        x = x * weight + self.bias
        return x


@saveable("flarejax.RMSNorm")
class RMSNorm(Module):
    """
    Scale the input by the reciprocal of the root mean square along the last
    axis. Then add a learnable offset and scale by a learnable weight along the
    last axis.

    Parameters
    ---
    eps: float
        Epsilon value for numerical stability.

    axis: int
        The axis to calculate the root mean square.
    """

    @typecheck
    def __init__(self, eps: float = 1e-6, axis: int = -1) -> None:
        self.eps = eps
        self.axis = axis

        self.weight = None
        self.bias = None

    def _build(self, x) -> None:
        self.weight = jnp.ones(x.shape[-1], dtype=x.dtype)
        self.bias = jnp.zeros(x.shape[-1], dtype=x.dtype)

    @jaxtyped(typechecker=typecheck)
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
