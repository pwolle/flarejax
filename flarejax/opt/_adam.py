import jax.numpy as jnp
from jaxtyping import Array, Float

from .._module import PathLookup
from .._serial import saveable
from ._opt import Optimizer

__all__ = [
    "Adam",
]


@saveable("falrejax.opt.Adam")
class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.eps_root = eps_root

        self.t = jnp.zeros((), jnp.uint64)
        self.m = {}
        self.v = {}

    def _build(self, key, x) -> None:
        if key not in self.m:
            self.m[key] = jnp.zeros_like(x)

        if key not in self.v:
            self.v[key] = jnp.zeros_like(x)

    def __call__(
        self,
        key: PathLookup,
        grad: Float[Array, "*s"],
    ) -> Float[Array, "*s"]:
        self.t = self.t + 1
        self._build(key, grad)

        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad**2

        m_hat = self.m[key] / (1 - self.beta1**self.t)
        v_hat = self.v[key] / (1 - self.beta2**self.t)

        grad_new = m_hat / (jnp.sqrt(v_hat + self.eps_root) + self.eps)
        return self.learning_rate * grad_new
