# FlareJax
FlareJax is a Python library for building neural networks and optimizers in Jax. It is designed to minimize the time between a new research idea and its implementation.

## Features
- Mutable modules for quick and dirty modifications via `Module`
- Serialization of modules via `@saveable`, `save`, and `load`
- Systematically modifying modules by using `flatten` and `unflatten`
- Safely handling shared/cyclical references and static arguments through `filter_jit`
- Commonly used NN layers and optimizers are included
- As a small codebase, it is relatively easy to understand and extend

## Quick Example
Define new Modules by subclassing `Module`. All methods are callable PyTrees.

```python
@flr.saveable("Example:Linear")  # optional, make saveable
class Linear(flr.Module):
    def __init__(self, key: PRNGKeyArray, dim: int):
        self.dim = dim
        self.w = None

    def __call__(self, x):
        # lazy initialization dependent on the input shape
        if self.w is None:
            self.w = jrn.normal(key, (x.shape[-1], self.dim))

        return x @ self.w

layer = Linear(jrn.PRNGKey(0), 3)
x = jnp.zeros((1, 4))

# the model is initialized after the first call
y = layer(x)
assert layer.w.shape == (4, 3)
```

For optimization, define a loss function, which takes the module as the first argument.

```python
def loss_fn(module, x, y):
    return jnp.mean((module(x) - y) ** 2)

opt = flr.opt.Adam(3e-4)

# automatically just-in-time compiled
opt, model, loss = flr.train(opt, model, loss_fn, x, y)
```

Models can be saved and loaded.

```python
flr.save(layer, "model.npz")

# load the model
layer = flr.load("model.npz")
assert isinstance(layer, Linear)
```

## Installation
FlareJax can be installed via pip. It requires Python 3.10 or higher and Jax 0.4.33 or higher.

```bash
pip install flarejax
```

## See Also
- [Jax Docs](https://jax.readthedocs.io/en/latest/): Jax is a library for numerical computing that is designed to be composable and fast.
- [Equinox library](https://github.com/patrick-kidger/equinox): FlareJax is heavily inspired by this awesome library.
- [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html): Many of the principles of mutability are inspired by PyTorch's `torch.nn.Module`.
- [NNX Docs](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html/): NNX is a library for neural networks in flax that also supports mutability.
- Always feel free to reach out to me via [email](mailto:paul.wollenhaupt@gmail.com).
