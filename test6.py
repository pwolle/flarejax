# %%
import jax
import jax.numpy as jnp


def f(x):
    return jnp.stack([x, x], 0)


def g(x, a):
    return jax.vmap(f, in_axes=a)(x)


x = jnp.zeros((2, 3))

jax.eval_shape(lambda x: g(x, 0), x)
