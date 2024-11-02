import jax.numpy as jnp


def make_test_pytree(num_els: int):
    return dict(
        a=jnp.arange(num_els*2).reshape(num_els, -1),
        b=jnp.arange(num_els*3).reshape(num_els, -1),
    )