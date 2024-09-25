import abc
import copy
from typing import Any, Callable, Self, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .._filter import filter_jit
from .._module import Module, PathLookup, flatten, unflatten
from .._tcheck import typecheck
from .._serial import saveable

__all__ = [
    "Optimizer",
]

T = TypeVar("T", bound=Module | list | tuple | dict)


@typecheck
def _filter_and_reconstruct(
    module: T,
    include: Callable[[PathLookup, Array], bool],
) -> tuple[
    dict[PathLookup, Float[Array, "..."]],
    Callable[[dict[PathLookup, Float[Array, "..."]]], T],
]:
    flat = flatten(module)

    active = {}
    static = {}

    for key, value in flat.items():
        if include(key, value):
            active[key] = value
            continue

        static[key] = value

    def reconstruct(active):
        return unflatten({**static, **active})

    return active, reconstruct


@typecheck
def _loss_gradient(
    loss: Callable[..., Float[Array, ""]],
    model: T,
    *args: Any,
    include: Callable[[PathLookup, Array], bool] = lambda *_: True,
) -> tuple[
    T,
    dict[PathLookup, Float[Array, "..."]],
    Float[Array, ""],
]:
    def include_(key, value):
        if not isinstance(value, jax.Array):
            return False

        if not jnp.issubdtype(value.dtype, jnp.floating):
            return False

        result = include(key, value)
        return result

    params, reconstruct = _filter_and_reconstruct(
        model,
        lambda k, v: include(k, v) and include_(k, v),
    )

    def loss_filtered(params):
        model = reconstruct(params)
        return loss(model, *args).mean(), model

    grad_func = jax.value_and_grad(loss_filtered, has_aux=True)
    (loss_val, model), grads = grad_func(params)
    return model, grads, loss_val


@saveable("flarejax.opt.Optimizer")
class Optimizer(Module):
    @filter_jit
    def minimize(
        self,
        loss: Callable[..., Float[Array, ""]],
        model: T,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Self, T, Float[Array, ""]]:
        return self.minimize_no_jit(loss, model, *args, **kwargs)

    @property
    def _include(self):
        if not hasattr(self, "include"):
            return lambda *_: True

        return self.include  # type: ignore

    def minimize_no_jit(
        self,
        loss: Callable[..., Float[Array, ""]],
        model: T,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Self, T, Float[Array, ""]]:
        model, grads, loss_val = _loss_gradient(
            loss,
            model,
            *args,
            **kwargs,
            include=self._include,
        )

        grads = self._call_map(grads)
        flat = flatten(model)

        for key, value in grads.items():
            flat[key] = flat[key] - value

        model = unflatten(flat)
        return self, model, loss_val

    @typecheck
    def _call_map(
        self, grads: dict[PathLookup, Float[Array, "..."]]
    ) -> dict[PathLookup, Float[Array, "..."]]:
        # shallow copy to avoid modifying the original dictionary
        grads = copy.copy(grads)

        for key, value in grads.items():
            grads[key] = self(key, value)

        return grads

    @abc.abstractmethod
    def __call__(
        self,
        key: PathLookup,
        grad: Float[Array, "*s"],
    ) -> Float[Array, "*s"]:
        raise NotImplementedError


class Chain(Optimizer):
    @typecheck
    def __init__(
        self,
        *optimizers: Callable[
            [PathLookup, Float[Array, "*s"]],
            Float[Array, "*s"],
        ],
    ):
        self.optimizers = optimizers

    @typecheck
    def __call__(
        self,
        key: PathLookup,
        grad: Float[Array, "*s"],
    ) -> Float[Array, "*s"]:
        for opt in self.optimizers:
            grad = opt(key, grad)

        return grad
