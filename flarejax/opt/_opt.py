"""
Tools for implementing new gradient based optimizers.
"""

import abc
import copy
from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, UInt32

from ..flr._filter import filter_jit
from ..flr._module import Module, flatten, unflatten
from ..flr._lookup import PathLookup, AttrLookup
from ..flr._tcheck import typecheck
from ..flr._serial import saveable

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


def _filter_frozen(module) -> Callable[[PathLookup, Array], bool]:
    """
    Create a function that returns True if the parameter should be included
    in the optimization step, by checking if any of the preceding modules in
    the depth first search have the frozen attribute set to True.
    """

    flat = flatten(module)
    frozen = set()

    # for performance, return a function that always returns True if there
    # are no frozen modules, this avoids iterating over all prefixes
    if not frozen:
        return lambda *_: True

    for key, value in flat.items():
        if key.path[-1] == AttrLookup("_frozen") and value is True:
            frozen.add(key.path[:-1])

    def include(key, value):
        path = key.path

        for i in range(len(path)):
            if path[:i] in frozen:
                return False

        return True

    return include


@typecheck
def _loss_gradient(
    loss: Callable[..., Float[Array, ""]],
    model: T,
    *args: Any,
) -> tuple[
    T,
    dict[PathLookup, Float[Array, "..."]],
    Float[Array, ""],
]:
    include_non_frozen = _filter_frozen(model)

    def include(key, value):
        if not isinstance(value, jax.Array):
            return False

        if not jnp.issubdtype(value.dtype, jnp.floating):
            return False

        if not include_non_frozen(key, value):
            return False

        return True

    params, reconstruct = _filter_and_reconstruct(
        model,
        lambda k, v: include(k, v) and include(k, v),
    )

    def loss_filtered(params):
        model = reconstruct(params)
        return loss(model, *args).mean(), model

    grad_func = jax.value_and_grad(loss_filtered, has_aux=True)
    (loss_val, model), grads = grad_func(params)
    return model, grads, loss_val


@saveable("flarejax.opt.Optimizer")
class Optimizer(Module):
    """
    Base class for implementing new gradient based optimizers.

    New optimizers should inherit from this class and implement the __call__
    method for transforming the gradients.
    The __call__ method expects a
    key of type PathLookup and a gradient array and should return an array of
    the same type.

    If any module or submodule that is being optimized has a frozen attribute
    set to True, the optimizer will not update the parameters of that module
    and all of its submodules (modules that occur later in a depth first
    search).
    """

    t: UInt32[Array, ""]

    @filter_jit
    def minimize(
        self,
        loss: Callable[..., Float[Array, ""]],
        model: T,
        *args: Any,
        **kwargs: Any,
    ) -> tuple["Optimizer", T, Float[Array, ""]]:
        """
        Apply the optimizer to the module for minimizing the loss function.

        Parameters
        ---
        loss: Callable[..., Float[Array, ""]]
            The loss function to minimize.
            The first argument should be the model.

        model: Module
            The model to optimize.

        args and kwargs: Any
            Additional arguments to pass to the loss function.


        Returns
        ---
        Self
            The optimizer with updated parameters.

        Module
            The updated model.

        Float[Array, ""]
            The value of the loss function.
        """

        return self._minimize_no_jit(loss, model, *args, **kwargs)

    def _minimize_no_jit(
        self,
        loss: Callable[..., Float[Array, ""]],
        model: T,
        *args: Any,
        **kwargs: Any,
    ) -> tuple["Optimizer", T, Float[Array, ""]]:
        model, grads, loss_value = _loss_gradient(
            loss,
            model,
            *args,
            **kwargs,
        )

        flat = flatten(model)
        params = {k: v for k, v in flat.items() if k in grads}

        grads = self(
            loss=loss_value,
            grads=grads,
            params=params,
        )

        for key, value in grads.items():
            flat[key] = flat[key] - value

        model = unflatten(flat)
        return self, model, loss_value

    def call_param(self, grad: Float[Array, "*s"], **_) -> Float[Array, "*s"]:
        return grad

    def call_model(
        self,
        grads: dict[PathLookup, Float[Array, "..."]],
        **_,
    ) -> dict[PathLookup, Float[Array, "..."]]:
        return grads

    def __call__(
        self,
        loss: Float[Array, ""],
        grads: dict[PathLookup, Float[Array, "..."]],
        params: dict[PathLookup, Float[Array, "..."]],
    ) -> dict[PathLookup, Float[Array, "..."]]:
        grads = self.call_model(
            loss=loss,
            params=params,
            grads=grads,
        )

        for key, grad in grads.items():
            grads[key] = self.call_param(
                key=key,
                grad=grad,
                param=params[key],
            )

        return grads


class Chain(Optimizer):
    """
    Combine multiple gradient transformations into a single optimizer.
    """

    def __init__(
        self,
        *optimizers: Optimizer,
    ) -> None:
        self.optimizers = optimizers

    def __call__(
        self,
        *,
        loss: Float[Array, ""],
        params: dict[PathLookup, Float[Array, "..."]],
        grads: dict[PathLookup, Float[Array, "..."]],
    ) -> dict[PathLookup, Float[Array, "..."]]:
        for opt in self.optimizers:
            grads = opt(
                loss=loss,
                params=params,
                grads=grads,
            )

        return grads
