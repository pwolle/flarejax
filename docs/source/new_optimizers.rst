Implementing Optimizers
#######################

Optimizers can be implemented by subclassing the ``flr.optim.Optimizer`` class.
This class offers four methods that can be overridden:

- ``__init__`` is used to initialize the hyperparameters of the optimizers.
- ``call_param`` is used for updating single parameters one at a time.
- ``call_model`` can be used for updating the the whole model
- ``__call__`` can be modified for advanced usage.

In the following we will go over the first three methods in more detail, by implementing the `Adam optimizer <https://arxiv.org/abs/1412.6980>`_.

The Adam Optimizer
==================

The Adam optimizer is a popular optimizer for training deep neural networks. It keeps track of an exponentially decaying average of past gradients and past squared gradients. The update rule is given by:

.. math::

	m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
	v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
	\delta \theta_t &= - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t

where :math:`\theta_t` is the parameter at time step :math:`t`,
:math:`\delta\theta_t` its update, :math:`g_t` its gradient, :math:`m_t` and :math:`v_t` are the first and second moment estimates, and :math:`\eta`, :math:`\beta_1`, :math:`\beta_2`, and :math:`\epsilon` are hyperparameters, that control the learning rate, the decay rate of the first moment estimate, the decay rate of the second moment estimate, and a small constant to prevent division by zero, respectively.


Initializing
------------
The ``__init__`` method is used to initialize the hyperparameters of the optimizer. In this case we will initialize the hyperparameters :math:`\eta`, :math:`\beta_1`, :math:`\beta_2`, and :math:`\epsilon`.

We will also initialize the first and second moment estimates :math:`m` and :math:`v` as empty dictionaries, and the time step :math:`t` as zero. 
The time must be stored as a Jax array, such that the minimization step will not be recompiled at each iteration.

.. code-block:: python

	class Adam(Optimizer):
		def __init__(self, η=0.001, β1=0.9, β2=0.999, ε=1e-8):
			self.η = η
			self.β1 = β1
			self.β2 = β2
			self.ε = ε
			
			self.m = {}
			self.v = {}
			self.t = jnp.zeros((), dtype=jnp.uint32)


Lazy EMAs
---------
When the model encounters a new parameter, the first and second moment estimates must be initialized. We will prepare a method for doing this lazily.

.. code-block:: python

	class Adam(Optimizer):
		...
			
		def _build(self, key):
			if key not in self.m:
				self.m[key] = jnp.zeros_like(x)

			if key not in self.v:
				self.v[key] = jnp.zeros_like(x)


Updating Parameters
-------------------
The ``call_param`` method is used for updating single parameters one at a time. We will implement the Adam update rule in this method.

.. code-block:: python

	class Adam(Optimizer):
		...
			
		def call_param(self, key, grad, **_):
			self._build(key)
			
			m = self.β1 * self.m[key] + (1 - self.β1) * grad
			v = self.β2 * self.v[key] + (1 - self.β2) * grad**2
			
			m_hat = m / (1 - self.β1**self.t)
			v_hat = v / (1 - self.β2**self.t)
			
			self.m[key] = m
			self.v[key] = v
			
			return self.η / (jnp.sqrt(v_hat) + self.ε) * m_hat


The ``call_param`` also recieves the value of the parameter, but we do not need it in this case. We will use the ``**_`` syntax to ignore it.


Updating the Timestep
---------------------
The time step :math:`t` is shared by all parameters, so we will update it in the ``call_model`` method.

.. code-block:: python

	class Adam(Optimizer):
		...
			
		def call_model(self, grads, **_):
			self.t += 1
			return grads


The ``call_model`` method also recieves the model parameters and the loss value as parameters, but we do not need them in this case. We will use the ``**_`` syntax to ignore them.


Putting it all together
-----------------------
The complete implementation of the Adam optimizer is shown below. The actual implementation in the library is not much longer, see :mod:`flarejax.opt.Adam` for the full implementation.

.. code-block:: python

	class Adam(Optimizer):
		def __init__(self, η=0.001, β1=0.9, β2=0.999, ε=1e-8):
			self.η = η
			self.β1 = β1
			self.β2 = β2
			self.ε = ε
			
			self.m = {}
			self.v = {}
			self.t = jnp.zeros((), dtype=jnp.uint32)

		def _build(self, key):
			if key not in self.m:
				self.m[key] = jnp.zeros_like(x)

			if key not in self.v:
				self.v[key] = jnp.zeros_like(x)

		def call_param(self, key, grad, **_):
			self._build(key)
			
			m = self.β1 * self.m[key] + (1 - self.β1) * grad
			v = self.β2 * self.v[key] + (1 - self.β2) * grad**2
			
			m_hat = m / (1 - self.β1**self.t)
			v_hat = v / (1 - self.β2**self.t)
			
			self.m[key] = m
			self.v[key] = v
			
			return self.η / (jnp.sqrt(v_hat) + self.ε) * m_hat

		def call_model(self, grads, **_):
			self.t += 1
			return grads

			

