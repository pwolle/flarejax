import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped

from .._module import PathLookup
from .._serial import saveable
from .._tcheck import typecheck
from ._opt import Optimizer

__all__ = [
    "Adam",
]


@saveable("falrejax.opt.Adam")
class Adam(Optimizer):
    """
    Implementation of the `Adam optimizer <https://arxiv.org/abs/1412.6980>`_.
    The Adam optimizer works by keeping a running exponential moving average of
    the first and second moments of the gradient, and using these to update the
    parameters.

    Parameters
    ---
    learning_rate: float
        The learning rate to use for the optimizer. A final scalar multiplier
        for the update step.

    beta1: float
        The exponential decay rate for the first moment estimate.

    beta2: float
        The exponential decay rate for the second moment estimate.

    eps: float
        Small value to prevent division by zero.

    eps_root: float
        Small value to prevent division by zero in the square root.
    """

    @typecheck
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

        self.t = jnp.zeros((), jnp.uint32)
        self.m = {}
        self.v = {}

    def _build(self, key, x) -> None:
        if key not in self.m:
            self.m[key] = jnp.zeros_like(x)

        if key not in self.v:
            self.v[key] = jnp.zeros_like(x)

    @jaxtyped(typechecker=typecheck)
    def __call__(
        self,
        key: PathLookup,
        grad: Float[Array, "*s"],
    ) -> Float[Array, "*s"]:
        self._build(key, grad)
        self.t = self.t + 1

        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad**2

        m_hat = self.m[key] / (1 - self.beta1**self.t)
        v_hat = self.v[key] / (1 - self.beta2**self.t)

        grad_new = m_hat / (jnp.sqrt(v_hat + self.eps_root) + self.eps)
        return self.learning_rate * grad_new
