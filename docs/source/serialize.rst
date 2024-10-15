Serialization
#############

Quickly saving and loading models including their hyperparameters, can be very helpful for research iteration speed.
Therefore FlareJax provides a simple way to do this with the ``@saveable`` decorator and the ``save`` and ``load`` functions.

Under the Hood
==============
FlareJax has its own functionality for flattening and unflattening modules with all their attributes into python dictionaries.
This mechanism is already a good start for serializing modules. However for some objects in the flattened dictionary serializing them safely is not trivial, since they might be module types or functions, which cannot be written to a ``.json`` or ``.npz`` file.

Therefore these objects need to be added to a global registry, that maps them to a unique string identifier. This identifier can be used to reconstruct the object when loading the module.
The ``@saveable`` decorator does exactly this. It registers the module in the global registry, which can be used to turn these objects to strings and back.

In the end, the ``save`` function simply flattens the module, converts all object that cannot are not arrays and cannot be serialized to ``json```directly to strings and saves the arrays with the ``json`` description of the remaining objects in a ``.npz`` file.

When using ``load`` the process is reversed. Here it is important to note that the global registry must contain all of the entries, which were used in the ``save`` process. Otherwise the loading process will not be able to convert the strings back to the original objects.

Example
=======
Here is a simple example of how the ``@saveable`` decorator can be used to mark the activation function as saveable, which allows for saving and loading the model.

.. code-block:: python

	activation = lambda x: jnp.maximum(0, x)
	flr.saveable()
	
	model = flr.net.Sequential(
		flr.net.Linear(key1, 1024),
		activation,
		flr.net.Linear(key2, 10),
		jnn.softmax
	)
	model(jnp.zeros((1, 784)))  # initialize the model

	# save the model
	flr.save("model.npz", model)

	# load the model
	model = flr.load("model.npz")


