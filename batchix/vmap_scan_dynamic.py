from typing import TypeVar, Callable, Tuple, Any

import chex
import jax
from einshape import jax_einshape as einshape
from jax import numpy as jnp
from jaxtyping import PyTree, Shaped, Array, Integer
from numpy.lib.index_tricks import IndexExpression

from batchix.batching import pytree_split_in_batches_with_remainder, pytree_sub_index_each_leaf, pytree_combine_batches
from batchix.tree_shape import pytree_get_shape_first_axis_equal

X = TypeVar("X")
Carry = TypeVar("Carry")
Y = TypeVar("Y")


def scan_batched_dynamic_padded(
    fn: Callable[[Carry, X], tuple[Carry, Y]],
    x: X,
    batch_size: int,
    fn_carry_init: Carry = None
) -> tuple[Carry, Y]:
    """
    Just as scan, apply fn to each of the input args (over the first batch axis).
    Allways split args into batches and loop over them.
    Not this function does not jit anything, you need to jit the given fn yourself.
    Note that the output first axis size of returned y will be equal to given x.
    :param fn: function to apply to each element. Returns: carry, y.
    :param x: the data to pass to fn, may be split into batches along first axis.
    :return: The result of fn of each batch element but all batches recombined (first axis is the batch axis).
    """
    # for too many elements we need to scan over batches
    x_batched, batch_remainder = pytree_split_in_batches_with_remainder(
        x,
        batch_size=batch_size,
        batch_remainder_strategy='Pad',
        rng_key=None
    )
    num_batches = pytree_get_shape_first_axis_equal(x_batched)

    # y for all batches
    y_all = None

    # go over batches
    carry = fn_carry_init
    for batch_i in range(num_batches):
        batch = pytree_sub_index_each_leaf(x_batched, jnp.s_[batch_i])
        carry, batch_y = fn(carry, batch)

        # create empty y_all
        if y_all is None:
            # create result tree with right sizes
            y_all = jax.tree_util.tree_map(
                lambda el: jnp.zeros(
                    (num_batches,)
                    + el.shape[:]
                ),
                batch_y,
            )
        y_all = jax.tree_util.tree_map(lambda l, b_y: l.at[batch_i].set(b_y), y_all, batch_y)

    y = pytree_combine_batches(y_all, batch_remainder)
    return carry, y
    



class DynamicVMapFlexibleLength:
    """
    Executed fn parallel of different batches of the input data input.
    This is eqal to a scan over batches, but the number of scan iterations is dynamic.
    :param fn:
    :param vmap_batch_size:
    """

    def __init__(self, fn: Callable[[X], Y], vmap_batch_size: int):
        self.fn = fn
        self.vmap_batch_size = vmap_batch_size
        self.fn_jit = jax.jit(self.fn)
        self.vmap_fn_jitted = None


    def __call__(self, x: X) -> Y:
        """
        Executed fn parallel of different batches of the input data input
        :param x: the input data (the first axis it the batch axis), the size of the first axis is flexible
        :return: y result of function batched in first axis
        """
        # manually cache the jitted function
        if self.vmap_fn_jitted is None:
            self.vmap_fn_jitted = jax.jit(jax.vmap(self.fn_jit))

        def fn(carry, x):
            return None, self.vmap_fn_jitted(x)

        carry, y = scan_batched_dynamic_padded(fn, x, batch_size=self.vmap_batch_size)
        return y
