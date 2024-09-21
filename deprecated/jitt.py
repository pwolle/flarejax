# %%
import importlib

import jax
import jax.numpy as jnp
import jax.random as jrn

import jax.tree_util as jtu
import flarejaxflex as fj

import functools

from tqdm import trange

importlib.reload(fj)


class Wrapper(fj.Module):
    def __init__(self, value):
        self.value = value


def jit(func):
    cache = {}

    @functools.partial(jax.jit, static_argnames=("struct_hash"))
    def compilable(leaves, struct_hash):
        arguments = jtu.tree_unflatten(cache[struct_hash], leaves)
        return func(*arguments.value[0], **arguments.value[1])

    def wrapper(*args, **kwargs):
        arguments = Wrapper((args, kwargs))

        leaves, struct = jtu.tree_flatten(arguments)
        struct_hash = hash(struct)

        if struct_hash not in cache:
            cache[struct_hash] = struct

        result = compilable(leaves, struct_hash)

    return wrapper


model = fj.Linear(jrn.PRNGKey(0), 1)
model(jnp.array([[1.0]]))


# @jax.jit
@jit
def identity(x):
    print("recompiling")
    return


for _ in range(4):
    leaves, struct = jtu.tree_flatten(model)

    # print(hash(struct))
    # print([(leaf.dtype, leaf.shape) for leaf in leaves])

    identity(model)
    # identity(leaves)
