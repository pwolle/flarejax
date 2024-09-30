"""
Linear transformations layers and similar modules.
"""

import jax.numpy as jnp
import jax.random as jrn
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from .._module import Module
from .._serial import saveable
from .._tcheck import typecheck

__all__ = [
    "Linear",
    "Bias",
    "Scale",
]


@saveable("flarejax.Linear")
class Linear(Module):
    """
    Apply a learnable linear transformation to last axis of the input.

    Parameters
    ---
    key: PRNGKeyArray
        Key to use for random initialization.

    dim: int
        Dimension of the output. The input dimension is inferred from the last
        axis of the first input.
    """

    weight: Float[Array, "dim_in dim"] | None

    def __init__(self, key: PRNGKeyArray, dim: int) -> None:
        self.key = key
        self.dim = dim

        self.weight = None

    def _build(self, x) -> None:
        if self.weight is not None:
            return

        dim_in = x.shape[-1]
        glorot = dim_in**-0.5

        self.weight = jrn.uniform(
            self.key,
            (dim_in, self.dim),
            dtype=x.dtype,
            minval=-glorot,
            maxval=+glorot,
        )

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim_in"],
    ) -> Float[Array, "*batch {self.dim}"]:
        self._build(x)

        assert self.weight is not None
        return x @ self.weight

    @property
    def dim_in(self) -> int | None:
        """
        Return the input dimension of the module. If the module does not have
        a fixed input dimension yet, return None.
        """
        if self.weight is None:
            return None

        return self.weight.shape[0]


@saveable("flarejax.Bias")
class Bias(Module):
    """
    Add a learnable bias to the last axis of the input.

    Parameters
    ---
    key: PRNGKeyArray
        Key to use for the initialization.
    """

    bias: Float[Array, "dim"] | None

    def __init__(self, key: PRNGKeyArray) -> None:
        self.key = key
        self.bias = None

    def _build(self, x) -> None:
        if self.bias is not None:
            return

        dim_in = x.shape[-1]
        glorot = dim_in**-0.5

        self.bias = jrn.uniform(
            self.key,
            (dim_in,),
            dtype=x.dtype,
            minval=-glorot,
            maxval=glorot,
        )

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        self._build(x)
        assert self.bias is not None

        return x + self.bias

    @property
    def dim(self) -> int | None:
        """
        Return the dimension of the bias. If the bias has not been initialized
        yet, return None.
        """
        if self.bias is None:
            return None

        return self.bias.shape[0]


@saveable("flarejax.Affine")
class Affine(Module):
    """
    Apply a learnable linear transformation followed by a learnable bias.

    Parameters
    ---
    key: PRNGKeyArray
        Key to use for random initialization.

    dim: int
        The output dimension of the linear transformation. The input dimension
        is inferred from the last axis of the first input.
    """

    def __init__(self, key: PRNGKeyArray, dim: int) -> None:
        self.linear = Linear(key, dim)
        self.bias = Bias(key)

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim_in"],
    ) -> Float[Array, "*batch dim"]:
        return self.bias(self.linear(x))

    @property
    def dim_in(self) -> int | None:
        return self.linear.dim_in


@saveable("flarejax.Scale")
class Scale(Module):
    """
    Scale the last axis of the input by a learnable vector.

    Parameters
    ---
    offset: bool
        If True the input will be scaled by `1 + scale` instead of `scale`.
        The scale is initialized to 0.
    """

    scale: Float[Array, "dim"] | None

    @typecheck
    def __init__(self, offset: bool = True) -> None:
        self.offset = offset
        self.scale = None

    def _build(self, x) -> None:
        if self.scale is not None:
            return

        self.scale = jnp.zeros((x.shape[-1],), x.dtype)

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        self._build(x)
        assert self.scale is not None

        if self.offset:
            return x * (1 + self.scale)

        return x * self.scale

    @property
    def dim(self) -> int | None:
        """
        Return the dimension of the scale. If the scale has not been initialized
        yet, return None.
        """
        if self.scale is None:
            return None

        return self.scale.shape[0]
