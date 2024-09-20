from typing import Callable, Any

import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrn
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from ._linear import Linear
from ._module import Module
from ._tcheck import typecheck


def _norm(x: Array, eps: float):
    v = jnp.square(x).sum()
    v = jnp.maximum(v, eps)

    r = lax.rsqrt(v)
    return x * r


def _softcap(x: Array, cap: float):
    x = x / cap
    x = jnp.tanh(x)
    x = x * cap
    return x


class DotProductAttention(Module):
    def __init__(
        self,
        scale: bool = True,
        tanh_cap: float | None = None,
        norm_qkv: bool = False,
        eps: float = 1e-4,
    ):
        self.scale = scale
        self.tanh_cap = tanh_cap
        self.norm_qkv = norm_qkv
        self.eps = eps

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
        if self.norm_qkv:
            q = _norm(q, self.eps)
            k = _norm(k, self.eps)
            v = _norm(v, self.eps)

        if self.scale:
            s = q.shape[-1] ** -0.5
            q = q * s

        att = jnp.einsum("...sd,...ad->...sa", q, k)

        if self.tanh_cap is not None:
            att = _softcap(att, self.tanh_cap)

        att = jnn.softmax(att, axis=-1)
        ret = jnp.einsum("...sa,...ad->...sd", att, v)

        return ret, att


_default_att_func = DotProductAttention()


class MultiHeadAttention(Module):
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
