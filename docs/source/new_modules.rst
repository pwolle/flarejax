Implementing Modules
####################

New modules can be implemented by subclassing ``flr.net.Module`` class. 
All methods of Modules will be treated as callable PyTrees.

In the following example we implement a simple linear layer. 
The layer taks in a vector(s) with of dimensionality ``dim_in`` and returns a vector(s) of dimensionality ``dim``.

The ``__init__`` method needs a key for the random initialization of the weights.

.. code-block:: python

	class Linear(flr.Module):
		def __init__(self, key: PRNGKeyArray, dim_in: int, dim: int):
			self.dim = dim
			self.dim_in = dim_in
			self.w = jrn.normal(key, (x.shape[-1], self.dim)) * 0.02

		def __call__(
			self,
			x: Float[Array, "*batch dim_in"],
		) -> Float[Array, "*batch dim"]:
			self._build(x)
			return x @ self.w


Lazy Initialization
===================
The weights of the layer do not need to be initialized in the ``__init__`` method. 
It might be convinient to initialize the weights only after the first call of the layer.
This has the benefit, that the value ``dim_in`` can be inferred from the input shape.

.. code-block:: python

	class LazyLinear(flr.Module):
		def __init__(self, key: PRNGKeyArray, dim: int):
			self.dim = dim
			self.w = None

		def __call__(
			self,
			x: Float[Array, "*batch dim_in"],
		) -> Float[Array, "*batch dim"]:
			if self.w is None:
				self.w = jrn.normal(key, (x.shape[-1], self.dim)) * 0.02
			return x @ self.w


Best Practices
==============
In the FlareJax codebase it is typically recommended to use the following best practices:
- Seperate the forward pass from the initialization of the weights by implementing a ``_build`` method.
- Add properties to the module for the inferred shapes of the input and output.

.. code-block:: python

	class BestLinear(flr.Module):
		def __init__(self, key: PRNGKeyArray, dim: int):
			self.dim = dim
			self.w = None

		def _build(self, x: Float[Array, "*batch dim_in"]):
			self.w = jrn.normal(key, (x.shape[-1], self.dim)) * 0.02

		def __call__(
			self,
			x: Float[Array, "*batch dim_in"],
		) -> Float[Array, "*batch dim"]:
			self._build(x)
			return x @ self.w

		@property
		def dim_in(self) -> int | None:
			if self.w is None:
				return None
			return self.w.shape[0]

			
Composing Modules
=================
Modules can have other modules as attributes. These submodules will automatically be treated as part of the PyTree.
However be carefull not to share the same key for the initialization, unless you want to initialize the weights of the submodules with the same values.

.. code-block:: python

	class MLP(flr.Module):
		def __init__(
			self, 
			key: PRNGKeyArray,
			dim_mid: int,
			dim_out: int
		):
			self.l1 = Linear(key, dim_mid)
			self.l2 = Linear(key, dim_out)

		def __call__(
			self,
			x: Float[Array, "*batch dim_in"],
		) -> Float[Array, "*batch dim_out"]:
			x = self.l1(x)
			x = jnn.relu(x)
			x = self.l2(x)
			return x


Serialization
=============
In order to save a module to a file, the module needs to be registered with the ``flr.saveable`` decorator.
If all leaves of the module are serializable, the module can be saved to a file with the ``flr.save`` function.
Serializable types are:

- Jax and Numpy arrays
- Floats, Ints, Bools, Strings and None
- Values registered with ``flr.saveable``

Some commonly used functions, like the activation functions from ``jax.nn`` are already registered with ``flr.saveable``.

.. code-block:: python
	
	# make the module saveable
	flr.saveable("Example:BestLinear")

	layer = BestLinear(jrn.PRNGKey(0), 3)
	x = jnp.zeros((1, 4))
	y = layer(x)
	assert layer.w.shape == (4, 3) 

	flr.save(layer, "model.npz")

	layer = flr.load("model.npz")
	assert isinstance(layer, BestLinear)