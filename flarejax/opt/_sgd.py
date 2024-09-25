from jaxtyping import Array, Float

from ._opt import Optimizer

__all__ = [
    "SGD",
]


class SGD(Optimizer):
    def __init__(
        self,
        learning_rate: float,
    ) -> None:
        self.lr = learning_rate

    def __call__(self, _, grad: Float[Array, "*s"]) -> Float[Array, "*s"]:
        return grad * self.lr
