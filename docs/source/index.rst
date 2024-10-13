Welcome to FlareJax's documentation!
####################################

.. toctree::
	:maxdepth: 2
	:hidden:
	:caption: The FlareJax Api

	flarejax
	net
	opt


	
.. toctree::
	:maxdepth: 2
	:hidden:
	:caption: Extending FlareJax

	new_modules
	new_optimizers

.. toctree::
	:maxdepth: 2
	:hidden:
	:caption: Under the Hood

	pytrees_refs
	serialize


FlareJax is a Python library for building neural networks and optimizers in Jax.
It is designed to minimize the time between a new research idea and its implementation.
Features include:

- Mutable modules for quick and dirty modifications via ``Module``
- Serialization of modules via ``@saveable, save`` and ``load``
- Systematically modifying modules by using ``flatten`` and ``unflatten``
- Safely handling shared/cyclical references and static arguments through ``filter_jit``
- Commonly used NN layers and optimizers are included
- As a small codebase, it is relatively easy to understand and extend

Installation
=============
FlareJax can be installed via pip. It requires Python 3.10 or higher and Jax 0.4.33 or higher.

.. code-block:: bash

	pip install flarejax


Quick Example
=============
Defining a neural network module is as simple as subclassing ``Module`` and implementing the ``__call__`` method.
Of course other methods can be implemented as well and will also be treated as callable PyTrees.

.. code-block:: python

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


Optimizing the paramteres is as simpel as defining a loss function, an optimizer and calling ``opt.minimize``.

.. code-block:: python

	# the loss function must takes the model as first argument
	def loss_fn(model, x, y):
		return jnp.mean((model(x) - y) ** 2)

	x = jnp.zeros((1, 4))
	y = jnp.zeros((1, 3))

	opt = flr.opt.Adam(3e-4)
	
	# automatically jit-ed
	opt, model, loss = opt.minimize(loss, layer, x, y)  


Saving the model to a ``.npz`` file can be done by calling ``save``.

.. code-block:: python

	flr.save(layer, "model.npz")

	# load the model
	layer = flr.load("model.npz")
	assert isinstance(layer, Linear)


Api
===
The following modules are available:

- :mod:`flarejax` for core functionality, including the ``Module`` base class
- :mod:`net` for neural network layers and utilities
- :mod:`opt` for optimizers, includes the ``Optimizer`` base class


See also
========
- Jax is a library for numerical computing that is designed to be composable and fast. `Jax Docs <https://jax.readthedocs.io/en/latest/>`_
- FlareJax is heavily inspired by the awesome `Equinox library <https://github.com/patrick-kidger/equinox>`_
- Many of the principles of mutability are inspired by PyTorch's `torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_
- NNX is a library for neural networks in flax that also supports mutability. `NNX Docs <https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html/>`_
- Always feel free to reach out to me via `email <mailto:paul.wollenhaupt@gmail.com>`_
