# FlareJax
Simple pytree module classes for Jax, strongly inspired by [Equinox](https://github.com/patrick-kidger/equinox)
- Referential transparency via strict immutability
- Easy manipulation using `.at` & `.set`
- Safe serialization including hyperparameters
- Bound methods and function transformations are also modules
- Auxillary information in key paths for filtered transformations


## Quick Examples
Modules work similar to dataclasses, but with the added benefit of being pytrees. Making them compatible with all Jax function transformations.
```python
import flarejax as fj

class Linear(fj.Module):
    # The __init__ method is automatically generated
    w: jax.Array
    b: jax.Array

    # only jax arrays and modules are considered pytree leaves
    aux: None = None

    # additional intialization methods via classmethods
    @classmethod
    def init(cls, key, dim_in, dim):
        w = jax.random.normal(key, (dim, dim_in)) * 0.02
        b = jax.numpy.zeros((dim,))
        return cls(w=w, b=b)

    def __call__(self, x):
        return self.w @ x + self.b

key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)

model = fj.Sequential(
    (
        Linear.init(key1, 3, 2),
        Linear.init(key2, 2, 5),
    )
)
```

Although modules are immutable, modified copies can be created using the `at` property.
```python
w_new = jax.numpy.ones((2, 3))
model = model.at[0].w.set(w_new)
```

The model can be serialized and deserialized using `fj.save` and `fj.load`.
```python
fj.save("model.npz", model)
model = fj.load("model.npz")
```

Flarejax includes wrappers of the Jax function transformations, which return callable modules.
```python
model = fj.VMap(model)
model = fj.Jit(model)
```

## Installation
Memmpy can be installed directly from PyPI using `pip`. It requires Python 3.10+ and Jax 0.4.26+.
```bash
pip install flarejax
```

## Design
Flarejax modules sacrifice some flexibility for the sake of a unified interface and safety. Flarejax code should alway be easy to reason about and should not contain any footguns from using python magic.
1. Everything is immutable and 
2. module fields can be either jax arrays, other modules or json-like data.

This makes it harder to use other jax libraries in flarejax modules. It is recommended to wrap the needed functionality in a module.
Most jax libraries should be compatible with flarejax modules, since they are simply callable pytrees.

## Roadmap
- [ ] Filtered grad transformation based on key paths
- [ ] Pretty printing for modules

## See also
- The beautiful [Equinox](https://github.com/patrick-kidger/equinox) library
