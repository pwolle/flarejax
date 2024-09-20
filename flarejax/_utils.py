import jax


def array_str(x: jax.Array, /) -> str:
    dtype = x.dtype.str[1:]
    shape = list(x.shape)

    head = f"{dtype}{shape}"
    return head
