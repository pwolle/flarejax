# %%
import importlib

import jax
import jax.numpy as jnp
import jax.random as jrn
import matplotlib.pyplot as plt
import jax.tree_util as jtu
import flarejax as fj

from tqdm import trange

importlib.reload(fj)

# create a simpel 1d regression dataset
key = jrn.PRNGKey(0)
key_x, key_y, key_model = jrn.split(key, 3)

n = 127
x = jrn.normal(key_x, (n, 1))
y = 2 * x + jrn.normal(key_y, (n, 1)) * 0.1

# plt.scatter(x, y)

model = fj.Linear(key_model, 1)
# model = jrn.normal(key_model, (1, 1024 * 32))

x_hat = jnp.linspace(-3, 3, 100)[:, None]
model(x_hat)


def loss(model, x, y):
    # y_hat = (model * x).sum(axis=-1, keepdims=True)
    y_hat = model(x)
    return ((y_hat - y) ** 2).mean()


# @jax.jit
# def sgd_step(model, x, y):
#     grad = jax.grad(loss)(model, x, y)
#     return model - 0.5 * grad


opt = fj.SGD(0.5)
# minimize = jax.jit(opt.minimize)


@jax.jit
def identity(x):
    print("recompiling")
    return x


for _ in range(4):
    # model = sgd_step(model, x, y)
    # model, _ = opt.minimize(loss, model, x, y)

    leaves, struct = jtu.tree_flatten(model)
    print(hash(struct))
    print([(leaf.dtype, leaf.shape) for leaf in leaves])

    # struct

    # print(hash(struct), hash(opt), hash(loss))
    model = identity(model)


# y_hat = model(x_hat)
# plt.plot(x_hat, y_hat, color="green")

# model.weight
