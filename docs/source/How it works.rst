How it works
############

Jax PyTrees
===========
The Jax ecosystem is built around the concept of PyTrees, which are nested structures of Python objects.
These objects can be anything from simple numbers to complex neural network modules.
For interfacing with Jax (i.e. ``jax.jit``) it is necessary to define a way to flatten these PyTrees into a list of arrays and some auxiliary data with which the original object can be reconstructed.

.. code-block:: python

    x = {"a": 1, "b": {"c": 2, "d": 3}}

    flat, aux = jtu.tree_flatten(x)
    print(f"Data: {flat}, Structure: {aux}")


Which gives us ``flat```which is an array of values that Jax hopefully understands and ``aux`` which a is a hashable object, that describes how to reconstruct the original PyTree.

.. code-block:: none
    :caption: Output

    Data: [1, 2, 3], Structure: PyTreeDef({'a': *, 'b': {'c': *, 'd': *}})


FlareJax Flattening
===================
Since FlareJax supports mutability with shared or even cyclical references, the flattening and unflattening process has to be a bit more involved than in other libraries. This however also opens up new possibilities for modifying modules in a systematic way.


Shared References
-----------------
Here is what might go wrong with the tree based flattening and unflattening process, when encountering shared references:

.. code-block:: python

    x = [1, 2]  # a mutable object
    y1 = {"a": x, "b": x}

    assert y1["a"] is y1["b"]  # shared reference

    leaves, aux = jtu.tree_flatten(y1)
    print(f"Leaves: {leaves}, Structure {aux}")

    y2 = jtu.tree_unflatten(aux, flat)

    y1["a"][0] = 3
    print(f"Modified y1 {y1}")

    y2["a"][0] = 3
    print(f"Modified y2 {y2}")


Here after flattening the outer part of ``y1``, i.e. the dictionary, the two arrays are flattened separately, although they are the same object.
When modifying the reconstructed object ``y2`` the shared reference is broken and the modification is only applied to one of the two arrays.

.. code-block:: none
    :caption: Output

    Leaves: [1, 2, 1, 2], Structure PyTreeDef({'a': [*, *], 'b': [*, *]})
    Modified y1 {'a': [3, 2], 'b': [3, 2]}
    Modified y2 {'a': [3, 2], 'b': [1, 2]}


To handle shared references, FlareJax implements its own flattening and unflattening process.

.. code-block:: python

    flat = flr.flatten(y1)
    pprint(flat)

    y2 = flr.unflatten(flat)
    
    y2["a"][0] = 4
    print(f"Modified y2 {y2}")


The flattened representation is a dictionary with keys describing where each value should be placed in the original PyTree. Keys, which end in ``.__class__`` describe the type of object at that position.
If a shared reference is encountered, the value is replaced by a description of the path to the first occurrence of the object.
The reconstructed object has the same shared references as the original object.

.. code-block:: none
    :caption: Output

    {obj[a][1]: 2,
    obj[b]: obj[a],
    obj[a].__class__: <class 'list'>,
    obj[a][0]: 1,
    obj.__class__: <class 'dict'>}
    Modified y2 {'a': [4, 2], 'b': [4, 2]}


Cyclical References
-------------------
If an object or its children contains a reference to itself, the tree based flattening and unflattening process will throw an Error, since it would get back to the same object infinitely often.


.. code-block:: python

    x = [1, 2]
    x.append(x)
    print(x)

    try:
        flat, aux = jtu.tree_flatten(x)
    except Exception:
        print("Error: Circular reference")


.. code-block:: none
    :caption: Output

    [1, 2, [...]]
    Error: Circular reference


The above describe approach of FlareJax flattening and unflattening process can handle cyclical references in the exact same way as shared references.

.. code-block:: python

    flat = flr.flatten(x)
    pprint(flat)

    y = flr.unflatten(flat)
    print(y)


.. code-block:: none
    :caption: Output

    {obj[2]: obj,
    obj[1]: 2,
    obj[0]: 1,
    obj.__class__: <class 'list'>}
    [1, 2, [...]]


Just in time Compilation with ``filter_jit``
============================================
Jax provides just in time compilation of functions with the ``jax.jit`` transformation.
To enable the compilation all of the inputs have to be converted into a representation that can be compiled by Jax.
The intermediate representation consists of a list of arrays/numbers and some hashable object, which describes their structure. If the structure changes between calls, the function is simply recompiled.

By default the PyTree API is used to perform this conversion. This gives us two options: Either we restrict the leaves of the PyTree to only be valid data types or we mark them as static afterwards. The first option is unfortunate, since the PyTree API can be usfull for modifying objects in a systematic way and the second option can be cumbersome.

Since FlareJax already introduces its own flattening and unflattening process, it can also has a new version of ``jax.jit`` that automatically treats all objects, which are not ``jax.Array``s as static arguments and tracks alls shared & cyclical references of the input, even between different arguments.

.. code-block::
    def func(v):
        v["a"][0] += 1
        return v

    x = [jnp.zeros(())]
    y = {"a": x, "b": x}  # shared reference

    print(jax.jit(func)(y))
    print(flr.filter_jit(func)(y))


.. This is unfortunate, since the PyTree API can also be very usefull for modifying objects in a systematic way and restricting the leaves to only be valid data types can be very limiting and the alternative of marking static arguments can be cumbersome.


.. Therefore, when using ``jax.jit`` all inputs have to be PyTrees with only valid data types  as leaves, valid data types themselves or have to be marked as static arguments.
.. By default the PyTree API is used 


.. This is very unfortunate, since the PyTree API can also be very usefull for modifying objects in a systematic way and restricting the leaves to only be valid data types can be very limiting.



.. Since ``jax.jit`` uses 




.. FlareJax provides a helper function ``filter_jit``, which automatically treats all objects, which are not ``jax.Array``s as static arguments and tracks alls shared & cyclical references of the input, even between different arguments.
