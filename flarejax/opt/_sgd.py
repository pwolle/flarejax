from jaxtyping import Array, Float, jaxtyped

from .._tcheck import typecheck
from ._opt import Optimizer

__all__ = [
    "SGD",
]


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) is arguably the simplest gradient-based
    optimization algorithm.
    The SGD optimizer works by updating the parameters in the opposite direction
    of the gradient, i.e. the best direction given a local linear approximation,
    scaled by a learning rate.

    Parameters
    ---
    learning_rate: float
        The learning rate to use for the optimizer.
    """

    @typecheck
    def __init__(
        self,
        learning_rate: float,
    ) -> None:
        self.lr = learning_rate

    @jaxtyped(typechecker=typecheck)
    def __call__(self, _, grad: Float[Array, "*s"]) -> Float[Array, "*s"]:
        return grad * self.lr
