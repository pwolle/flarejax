import flarejax as flr
import jax.numpy as jnp


def func(v):
    # print(v)
    v["a"][0] += 1
    # print(v)
    return v


x = [jnp.zeros(())]
y = {"a": x, "b": x}  # shared reference
# y = [x, x]

# print(func(y))
# print(jax.jit(func)(y))
print(flr.filter_jit(func)(y))