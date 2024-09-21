import jax.numpy as jnp
import jax.random as jrn
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from ._module import Module
from ._tcheck import typecheck

__all__ = [
    "Linear",
    "Bias",
    "Scale",
]


class Linear(Module):
    weight: Float[Array, "dim_in dim"] | None

    def __init__(self, key: PRNGKeyArray, dim: int) -> None:
        self.key = key
        self.dim = dim

        self.weight = None

    def _build(self, x) -> None:
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
        if self.weight is None:
            self._build(x)

        assert self.weight is not None
        return x @ self.weight

    @property
    def dim_in(self) -> int | None:
        if self.weight is None:
            return None

        return self.weight.shape[0]

    # def __repr__(self) -> str:
    #     head = f"{type(self).__name__}("
    #     body = f"dim_in={self.dim_in}, dim={self.dim})"
    #     tail = ")"
    #     return head + body + tail


class Bias(Module):
    bias: Float[Array, "dim"] | None

    def __init__(self, key: PRNGKeyArray) -> None:
        self.key = key
        self.bias = None

    def _build(self, x) -> None:
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
        if self.bias is None:
            self._build(x)

        assert self.bias is not None
        return x + self.bias

    @property
    def dim(self) -> int | None:
        if self.bias is None:
            return None

        return self.bias.shape[0]

    # def __repr__(self) -> str:
    #     head = f"{type(self).__name__}("
    #     body = f"dim={self.dim})"
    #     tail = ")"
    #     return head + body + tail


@jaxtyped(typechecker=typecheck)
class Scale(Module):
    scale: Float[Array, "dim"] | None

    def __init__(self, offset: bool = False) -> None:
        self.offset = offset
        self.scale = None

    def _build(self, x) -> None:
        self.scale = jnp.zeros((x.shape[-1],), x.dtype)

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        if self.scale is None:
            self._build(x)

        assert self.scale is not None

        if self.offset:
            return x * (1 + self.scale)

        return x * self.scale
