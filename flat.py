# %%
import flarejax as flr
import jax
import jax.numpy as jnp
import jax.random as jrn
import jax.nn as jnn
import jax.tree_util as jtu
from pprint import pprint

# %%
x = {"a": 1, "b": {"c": 2, "d": 3}}

flat, aux = jtu.tree_flatten(x)
print(f"Arrays: {flat}, Structure {aux}")

# %%

x = [1, 2]  # a mutable object
y1 = {"a": x, "b": x}

assert y1["a"] is y1["b"]  # shared reference

flat, aux = jtu.tree_flatten(y1)
print(f"Arrays: {flat}, Structure {aux}")

y2 = jtu.tree_unflatten(aux, flat)

y1["a"][0] = 0
print(f"Modified y1 {y1}")

y2["a"][0] = 0
print(f"Modified y2 {y2}")

# %%
x = [1, 2]  # a mutable object
y1 = {"a": x, "b": x}

flat = flr.flatten(y1)
pprint(flat)

y2 = flr.unflatten(flat)

y2["a"][0] = 4
print(f"Modified y2 {y2}")

# %%

x = [1, 2]
x.append(x)
print(x)

try:
    flat, aux = jtu.tree_flatten(x)
except Exception:
    print("Error: Circular reference")


# %%
# import importlib

# importlib.reload(flr)

import flarejax as flr
import jax.numpy as jnp


def func(v):
    print(v)
    v["a"][0] += 1
    # print(v)
    return v


x = [jnp.zeros(())]
y = {"a": x, "b": x}  # shared reference
# y = [x, x]

# print(jax.jit(func)(y))
print(flr.filter_jit(func)(y))

# %%
key = jrn.PRNGKey(0)
key1, key2 = jrn.split(key)

model = flr.net.Sequential(
    flr.net.Linear(key1, 128),
    jnn.relu,
    flr.net.Linear(key2, 10),
)

x = model(jnp.zeros((1, 784)))
