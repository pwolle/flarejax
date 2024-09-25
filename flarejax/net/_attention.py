"""
Attention mechanisms for neural networks.
"""

from typing import Any, Callable

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrn
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from ._linear import Linear
from .._module import Module
from .._serial import saveable
from .._tcheck import typecheck

__all__ = [
    "DotProductAttention",
    "MultiHeadAttention",
]


@saveable("flarejax.net.DotProductAttention")
class DotProductAttention(Module):
    """
    Module for computing 'Dot Product Attention' between query, key and value
    tensors.

    Parameters
    ---
    scale: bool = True
        Whether to use scaling by the square root of the dimension.
        Used in the original 'Attention is All You Need' paper [1].

    score: Callable[[Float[Array, "*s"]], Float[Array, "*s"]] | None = None
        Function to apply to the attention scores before softmax.

    dtype: Any = jnp.float32
        Data type to use for the attention scores. Even when using bfloat16
        training, it is recommended to use float32 for the attention scores.

    References
    - [1] https://arxiv.org/abs/1706.03762
    - [2] https://arxiv.org/abs/2102.01625
    """

    @typecheck
    def __init__(
        self,
        scale: bool = True,
        score: Callable[[Float[Array, "*s"]], Float[Array, "*s"]] | None = None,
        dtype: Any = jnp.float32,
    ):
        self.scale = scale
        self.score = score
        self.dtype = dtype

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        q: Float[Array, "*batch seq dim_head"],
        k: Float[Array, "*batch aux dim_head"],
        v: Float[Array, "*batch aux dim_v"],
    ) -> tuple[
        Float[Array, "*batch seq dim_v"],
        Float[Array, "*batch aux seq"],
    ]:
        if self.scale:
            s = q.shape[-1] ** -0.5
            q = q * s

        att = jnp.einsum("...hsd,...had->...hsa", q, k)

        if self.score is not None:
            att = self.score(att)

        att = jnn.softmax(att.astype(self.dtype), axis=-1).astype(att.dtype)
        ret = jnp.einsum("...hsa,...had->...hsd", att, v)

        return ret, att


_default_att_func = DotProductAttention()


@saveable("flarejax.net.MultiHeadAttention")
class MultiHeadAttention(Module):
    """
    Module for computing 'Multi-Head Attention' between query-, key- and value
    tensors [1].

    Parameters
    ---
    key: PRNGKeyArray
        Random key for initializing the internal Linear modules.

    dim: int
        Dimension of the input and output tensors.

    dim_head: int
        Dimension of each head.

    att_func: Callable[[Float[Array, "*batch heads seq dim_head"],
                        Float[Array, "*batch heads aux dim_head"],
                        Float[Array, "*batch heads aux dim_v"]],
                        tuple[Float[Array, "*batch heads seq dim_v"], Any]]
        Attention function to use. Default is DotProductAttention.
        This function computes the attention between the query, key and value
        tensors and returns the output tensor and any other value like the
        attention weights that might be useful for debugging.

    References
    - [1] https://arxiv.org/abs/1706.03762
    """

    @typecheck
    def __init__(
        self,
        key: PRNGKeyArray,
        dim: int,
        dim_head: int,
        att_func: Callable[
            [
                Float[Array, "*batch heads seq dim_head"],
                Float[Array, "*batch heads aux dim_head"],
                Float[Array, "*batch heads aux dim_v"],
            ],
            tuple[Float[Array, "*batch heads seq dim_v"], Any],
        ] = _default_att_func,
    ) -> None:
        assert dim % dim_head == 0

        self.dim = dim
        self.dim_head = dim_head
        self.att_func = att_func

        key_qkv, key_out = jrn.split(key)

        self.qkv = Linear(key_qkv, dim * 3)
        self.out = Linear(key_out, dim)

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch seq dim_in"],
    ) -> Float[Array, "*batch seq {self.dim}"]:
        qkv = self.qkv(x)

        qkv = qkv.reshape((*qkv.shape[:-1], -1, self.dim_head * 3))
        q, k, v = jnp.split(qkv, 3, axis=-1)

        ret, _ = self.att_func(q, k, v)
        ret = ret.reshape((*ret.shape[:-2], -1))

        return self.out(ret)


@saveable("flarejax.net.CrossAttention")
class CrossAttention(Module):
    """
    Perform cross-attention between two sequences of vectors.

    Parameters
    ---
    key: PRNGKeyArray
        Random key for initializing the internal Linear modules.

    dim: int
        Dimension of the input and output tensors.

    dim_head: int
        Dimension of each head.

    att_func: Callable[[Float[Array, "*batch heads seq dim_head"],
                        Float[Array, "*batch heads aux dim_head"],
                        Float[Array, "*batch heads aux dim_v"],
                        Any],
                        tuple[Float[Array, "*batch heads seq dim_v"], Any]]
        Attention function to use. Default is DotProductAttention.
        This function computes the attention between the query, key and value
        tensors and returns the output tensor and any other value like the
        attention weights that might be useful for debugging.
    """

    @typecheck
    def __init__(
        self,
        key: PRNGKeyArray,
        dim: int,
        dim_head: int,
        att_func: Callable[
            [
                Float[Array, "*batch heads seq dim_head"],
                Float[Array, "*batch heads aux dim_head"],
                Float[Array, "*batch heads aux dim_v"],
            ],
            tuple[Float[Array, "*batch heads seq dim_v"], Any],
        ] = _default_att_func,
    ) -> None:
        assert dim % dim_head == 0

        self.dim = dim
        self.dim_head = dim_head
        self.att_func = att_func

        key_q, key_kv, key_out = jrn.split(key, 3)

        self.q = Linear(key_q, dim)
        self.kv = Linear(key_kv, dim * 2)

        self.out = Linear(key_out, dim)

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        x: Float[Array, "*batch seq dim_in"],
        y: Float[Array, "*batch aux dim_in"],
    ) -> Float[Array, "*batch seq {self.dim}"]:
        q = self.q(x)
        q = q.reshape((*q.shape[:-1], -1, self.dim_head))

        kv = self.kv(y)
        kv = kv.reshape((*kv.shape[:-1], -1, self.dim_head * 2))
        k, v = jnp.split(kv, 2, axis=-1)

        ret, _ = self.att_func(q, k, v)
        ret = ret.reshape((*ret.shape[:-2], -1))

        return self.out(ret)
