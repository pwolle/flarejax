"""
Wrapper around the populer einops library.
"""

from typing import Literal

import einops

from .._module import Module
from .._tcheck import typecheck

__all__ = [
    "Rearrange",
    "Reduce",
    "Repeat",
    "Einsum",
]


@typecheck
class Rearrange(Module):
    """
    Rearrange the input tensor according to the given pattern.
    See `einops.rearrange` for more information.

    Parameters
    ---
    pattern: str
        The pattern to rearrange the input tensor.

    dims: int
        The dimensions to use in the pattern.
    """

    def __init__(self, pattern: str, **dims: int) -> None:
        self.pattern = pattern
        self.dims = dims

    def __call__(self, *args):
        return einops.rearrange(*args, pattern=self.pattern, **self.dims)


@typecheck
class Reduce(Module):
    """
    Reduce the input tensor according to the given pattern and the reduction
    type. See `einops.reduce` for more information.

    Parameters
    ---

    pattern: str
        The pattern to reduce the input tensor.

    reduction: str
        The reduction type to use. Can be one of "sum", "mean", "max", "min",
        or "prod".

    dims: int
        The dimensions to use in the pattern.
    """

    def __init__(
        self,
        pattern: str,
        reduction: Literal["sum", "mean", "max", "min", "prod"],
        **dims,
    ) -> None:
        self.pattern = pattern
        self.reduction = reduction
        self.dims = dims

    def __call__(self, *args):
        return einops.reduce(
            *args,
            pattern=self.pattern,
            reduction=self.reduction,
            **self.dims,
        )


@typecheck
class Repeat(Module):
    """
    Repeat the input tensor according to the given pattern. See `einops.repeat`
    for more information.

    Parameters
    ---
    pattern: str
        The pattern to repeat the input tensor.

    dims: int
        The dimensions to use in the pattern.
    """

    def __init__(self, pattern: str, **dims: int) -> None:
        self.pattern = pattern
        self.dims = dims

    def __call__(self, *args):
        return einops.repeat(*args, pattern=self.pattern, **self.dims)


@typecheck
class Einsum(Module):
    """
    Wrapper around `einops.einsum`. Performs a contraction on the input tensors
    according to the given pattern.

    Parameters
    ---
    pattern: str
        The pattern to contract the input tensors.

    dims: int
        The dimensions to use in the pattern.
    """

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern

    def __call__(self, *args):
        return einops.einsum(*args, pattern=self.pattern)  # type: ignore
