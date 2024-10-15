import jax.numpy as jnp

from ._opt import Optimizer


__all__ = [
    "Trapezoid",
]


class Trapezoid(Optimizer):
    def __ini__(
        self,
        total: int,
        warm: int,
        cool: int,
    ):
        self.total = total
        self.warm = warm
        self.cool = cool

        self.t = jnp.zeros((), jnp.uint32)

    def call_param(self, grad, **_):
        s = jnp.interp(
            self.t,
            xp=jnp.array([0, self.warm, self.total - self.cool, self.total]),
            fp=jnp.array([0, 1, 1, 0]),
        )
        return grad * s

    def call_model(self, grads, **_):
        if grads:
            self.t = self.t + 1

        return grads
