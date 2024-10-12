import jax.random as jrn
from jaxtyping import PRNGKeyArray

__all__ = [
    "RandomKey",
]


class RandomKey:
    """
    Simple wrapper around a PRNGkey that allows for easy splitting using the
    object state.
    """

    def __init__(self, key_or_seed: PRNGKeyArray | int):
        if isinstance(key_or_seed, int):
            key_or_seed = jrn.PRNGKey(key_or_seed)

        self._key = key_or_seed

    @property
    def key(self) -> PRNGKeyArray:
        """
        Get a new PRNGKeyArray each time this property is accessed.
        """
        self._key, key = jrn.split(self._key)
        return key

    def get(self, n: int) -> list[PRNGKeyArray]:
        """
        Get a list of n new PRNGKeyArrays.
        """
        self._key, *keys = jrn.split(self._key, n + 1)
        return keys
