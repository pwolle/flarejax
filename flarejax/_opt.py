from typing import Callable, Any, TypeVar
from jaxtyping import Float, Array

import functools
import jax
import jax.numpy as jnp

from ._module import Module, flatten, unflatten, PathLookup


T = TypeVar("T", bound=Module | list | tuple | dict)


def filter_module(
    module: T,
    include: Callable[[PathLookup, Array], bool],
) -> tuple[
    dict[PathLookup, Float[Array, "..."]],
    Callable[[dict[PathLookup, Float[Array, "..."]]], T],
]:
    flat, aux = flatten(module)

    active = {}
    static = {}

    for key, value in flat.items():
        if include(key, value):
            active[key] = value
            continue

        static[key] = value

    def reconstruct(active):
        flat = {**static, **active}
        return unflatten(flat, aux)

    return active, reconstruct


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

    params, reconstruct = filter_module(
        model,
        lambda k, v: include(k, v) and include_(k, v),
    )

    def loss_filtered(params):
        model = reconstruct(params)
        return loss(model, *args).mean(), model

    grad_func = jax.value_and_grad(loss_filtered, has_aux=True)
    (loss_val, model), grads = grad_func(params)
    return model, grads, loss_val


class Optimizer(Module):
    include: Callable[[PathLookup, Array], bool]

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.include = lambda *_: True
        return obj

    @functools.partial(jax.jit, static_argnames=("loss",))
    def minimize(
        self,
        loss: Callable[..., Float[Array, ""]],
        model: T,
        *args: Any,
    ) -> tuple[T, Float[Array, ""]]:
        print(model)

        model, grads, loss_val = _loss_gradient(
            loss,
            model,
            *args,
            include=self.include,
        )

        grads = self(grads)
        flat, aux = flatten(model)

        for key, value in grads.items():
            flat[key] = flat[key] - value

        model = unflatten(flat, aux)
        return model, loss_val

    def __call__(
        self,
        grads: dict[PathLookup, Float[Array, "..."]],
    ) -> dict[PathLookup, Float[Array, "..."]]:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float) -> None:
        self.lr = lr

    def __call__(self, grads) -> dict[PathLookup, Float[Array, "..."]]:
        for key, value in grads.items():
            grads[key] = value * self.lr

        return grads
