"""
Tools for implementing new gradient based optimizers.
"""

import abc
import copy
from typing import Any, Callable, Self, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped

from .._filter import filter_jit
from .._module import Module, flatten, unflatten
from .._lookup import PathLookup, AttrLookup
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

    @filter_jit
    def minimize(
        self,
        loss: Callable[..., Float[Array, ""]],
        model: T,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Self, T, Float[Array, ""]]:
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
            The optimizer.

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
    ) -> tuple[Self, T, Float[Array, ""]]:
        model, grads, loss_val = _loss_gradient(
            loss,
            model,
            *args,
            **kwargs,
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
    """
    Combine multiple gradient transformations into a signle optimizer.
    """

    def __init__(
        self,
        *optimizers: Callable[
            [PathLookup, Float[Array, "*s"]],
            Float[Array, "*s"],
        ],
    ) -> None:
        self.optimizers = optimizers

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        key: PathLookup,
        grad: Float[Array, "*s"],
    ) -> Float[Array, "*s"]:
        for opt in self.optimizers:
            grad = opt(key, grad)

        return grad
