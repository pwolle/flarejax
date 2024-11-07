"""
Attention mechanisms for neural networks.
"""

from typing import Literal

import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from ._linear import Linear
from ..flr._module import Module
from ..flr._serial import saveable
from ..flr._tcheck import typecheck


__all__ = [
    "CrossAttention",
    "DotProductAttention",
]


def normalize(x: Array, axis: int) -> Array:
    """
    Normalize the input tensor along the given axis.

    Parameters
    ---
    x: Array
        The input tensor.

    axis: int
        The axis along which to normalize the tensor.

    Returns
    ---
    Array
        The normalized tensor.
    """
    n = jnp.linalg.norm(x, axis=axis, keepdims=True)
    n = jnp.maximum(n, 1e-6)
    return x / n


def softcap(x: Array, cap: float) -> Array:
    """
    Apply a soft cap to the input tensor.

    Parameters
    ---
    x: Array
        The input tensor.

    cap: float
        The cap value.

    Returns
    ---
    Array
        The capped tensor.
    """
    return jnn.tanh(x / cap) * cap


@saveable("flarejax.net.DotProductAttention")
class DotProductAttention(Module):
    """
    Apply dot-product attention to a sequence.

    Parameters
    ---
    key: PRNGKeyArray
        The key to use for random number generation.

    dim: int
        The dimension of the final output.

    heads_q: int
        The number of heads for the query.

    heads_k: int | None
        The number of heads for the key. If None, defaults to `heads_q`.

    kwargs: Any
        For the full list of keyword arguments, see
        `jax.nn.dot_product_attention`.
    """

    @typecheck
    def __init__(
        self,
        key: PRNGKeyArray,
        dim: int,
        heads_q: int,
        heads_k: int | None = None,
        *,
        norm: bool = False,
        tanh_cap: float | None = None,
        scale: float | None = None,
        is_causal: bool = False,
        key_value_seq_lengths: Array | None = None,
        implementation: Literal["xla", "cudnn"] | None = None,
    ):
        heads_k = heads_k or heads_q

        assert dim % heads_q == 0
        assert dim % heads_k == 0
        assert heads_k % heads_q == 0

        self.dim = dim
        self.dim_head = dim // heads_q

        self.heads_q = heads_q
        self.heads_k = heads_k

        self.norm = norm
        self.tanh_cap = tanh_cap

        self.scale = scale
        self.is_causal = is_causal
        self.key_value_seq_lengths = key_value_seq_lengths
        self.implementation = implementation

        self.qkv = Linear(
            key,
            self.dim_head * heads_q + self.dim_head * heads_k * 2,
        )

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch seq dim"],
        bias: Array | None = None,
        mask: Array | None = None,
    ) -> Float[Array, "*batch seq dim"]:
        qkv = self.qkv(x)
        qkv = qkv.reshape((*qkv.shape[:-1], -1, self.dim_head))

        if self.norm:
            qkv = normalize(qkv, axis=-1)

        if self.tanh_cap is not None:
            qkv = softcap(qkv, self.tanh_cap)

        q = qkv[..., : self.heads_q, :]
        k = qkv[..., self.heads_q : self.heads_q + self.heads_k, :]
        v = qkv[..., self.heads_q + self.heads_k :, :]

        y = jnn.dot_product_attention(
            q,
            k,
            v,
            bias=bias,
            mask=mask,
            scale=self.scale,
            is_causal=self.is_causal,
            key_value_seq_lengths=self.key_value_seq_lengths,
            implementation=self.implementation,  # type: ignore
        )
        return y.reshape((*y.shape[:-2], -1))

    @property
    def dim_in(self) -> int | None:
        """
        Return the input dimension of the module. If the module does not have
        a fixed input dimension yet, return None.
        """
        return self.qkv.dim_in


@saveable("flarejax.net.Crossattention")
class CrossAttention(Module):
    """
    Perform cross-attention between two sequences.

    Parameters
    ---
    key: PRNGKeyArray
        The key to use for random number generation.

    dim: int
        The dimension of the final output.

    heads_q: int
        The number of heads for the query.

    heads_k: int | None
        The number of heads for the key. If None, defaults to `heads_q`.

    kwargs: Any
        For the full list of keyword arguments, see
        `jax.nn.dot_product_attention`.
    """

    @typecheck
    def __init__(
        self,
        key: PRNGKeyArray,
        dim: int,
        heads_q: int,
        heads_k: int | None = None,
        bias: Array | None = None,
        mask: Array | None = None,
        *,
        scale: float | None = None,
        is_causal: bool = False,
        key_value_seq_lengths: Array | None = None,
        implementation: Literal["xla", "cudnn"] | None = None,
    ):
        heads_k = heads_k or heads_q

        assert dim % heads_q == 0
        assert dim % heads_k == 0
        assert heads_k % heads_q == 0

        self.dim = dim
        self.dim_head = dim // heads_q

        self.heads_q = heads_q
        self.heads_k = heads_k

        self.bias = bias
        self.mask = mask
        self.scale = scale
        self.is_causal = is_causal
        self.key_value_seq_lengths = key_value_seq_lengths
        self.implementation = implementation

        self.q = Linear(key, self.dim_head * heads_q)
        self.kv = Linear(key, self.dim_head * heads_k * 2)

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch seq dim_seq"],
        y: Float[Array, "*batch aux dim_aux"],
    ) -> Float[Array, "*batch seq dim_seq"]:
        q = self.q(x)
        q = jnp.reshape(q, (*q.shape[:-1], self.heads_q, self.dim_head))

        kv = self.kv(y)
        kv = jnp.reshape(kv, (*kv.shape[:-1], self.heads_k * 2, self.dim_head))

        k = kv[..., : self.heads_k, :]
        v = kv[..., self.heads_k :, :]

        y = jnn.dot_product_attention(
            q,
            k,
            v,
            bias=self.bias,
            mask=self.mask,
            scale=self.scale,
            is_causal=self.is_causal,
            key_value_seq_lengths=self.key_value_seq_lengths,
            implementation=self.implementation,  # type: ignore
        )
        return y.reshape((*y.shape[:-2], -1))

    @property
    def dim_in_x(self) -> int | None:
        """
        Return the input dimension of the first sequence. If the module does not
        have a fixed input dimension yet, return None.
        """
        return self.q.dim_in

    @property
    def dim_in_y(self) -> int | None:
        """
        Return the input dimension of the second sequence. If the module does not
        have a fixed input dimension yet, return None.
        """
        return self.kv.dim_in
