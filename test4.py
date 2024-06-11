# %%
import jax.tree_util as jtu


def flatten_one_level_with_path(tree):
    return jtu.tree_flatten_with_path(
        tree,
        is_leaf=lambda x: x is not tree,
    )[0]


def is_leaf(x):
    return not isinstance(x, (list))


def dfs(t, path=()):
    """
    recursive left to right dfs using flatten_one_level_with_path
    """
    r = []

    for i, x in flatten_one_level_with_path(t):
        if is_leaf(x):
            r.append(((*path, (i, t)), x))
        else:
            r.extend(dfs(x, (*path, i, t)))

    return tuple(r)


def flatten_with_path(t):
    return dfs(t), jtu.tree_structure(t)


nested = [
    [
        1,
        2,
        3,
    ],
    [],
    [
        4,
        5,
        6,
        [
            7,
            8,
            9,
        ],
    ],
]

# dfs(nested), jtu.tree_flatten(nested)[0]
r, t = flatten_with_path(nested)
print(t)

for k, x in r:
    print(k, x)


t2, f = jtu.tree_flatten(nested)

jtu.tree_unflatten(f, list(map(lambda x: x[-1], r))) == nested
