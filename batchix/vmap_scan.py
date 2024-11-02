import math
from typing import TypeVar, Callable, Tuple, Any

import chex
import jax
from einshape import jax_einshape as einshape
from jax import numpy as jnp
from jaxtyping import PyTree, Shaped, Array, Integer
from numpy.lib._index_tricks_impl import IndexExpression

from batchix.batching import pytree_split_in_batches_and_padd, pytree_combine_batches
from batchix.tree_shape import pytree_get_shape_first_axis_equal, pytree_get_shape_first_n_equal

X = TypeVar('X')
Y = TypeVar('Y')


def vmap_batched_scan(
    fn: Callable[[...], Any] | Callable[[Any, ...], tuple[Any, Any]],
    args: tuple,
    batch_size: int,
    fn_has_carry: bool = False,
    fn_carry_init: Any = None,
    batch_axis_needs_to_be_dividable_by_batch_size=True
) -> Any | tuple[Any, Any]:
    """
    Just as normal vmap, apply fn to each of the input args (over the first batch axis).
    But if the batch axis size is larger than batch size, do scan over inner vmap (with batch_size).
    :param fn: function to apply to each element.
    :param args: the batched args to pass to fn.
    :param fn_has_carry: if true has signature fn(carry, *args) -> carry, y.
                            Where carry is a state that changes between the inner vmap calls.
                            if is true this function returns (carry, y).
    :param batch_axis_needs_to_be_dividable_by_batch_size: if true,
                args batch axis size needs to be dividable by batch size if args batch axis size < batch_size.
                Otherwise, may throw some padded values from vmap away.
    :return: The result of fn of each element (first axis is the batch axis).
    """
    num_elements = pytree_get_shape_first_axis_equal(args)
    def vmap_on_batch(carry, x):
        if not fn_has_carry:
            return None, jax.vmap(fn)(*x)
        else:
            carry, y = jax.vmap(fn, in_axes=(None, )+(0,)*len(args))(carry, *x)
            return carry, y

    # call with single vmap
    if num_elements <= batch_size:
        carry, y = vmap_on_batch(fn_carry_init, args)
        if fn_has_carry:
            return carry, y
        else:
            return y
    else:
        # for too many elements we need to scan over vmap -> otherwise too many elements for vmap
        args_batched, ignore_n_from_last_batch = pytree_split_in_batches_and_padd(
            args, batch_size=batch_size,
            fail_when_x_not_devidable_by_batch_size=batch_axis_needs_to_be_dividable_by_batch_size
        )
        carry, y = jax.lax.scan(vmap_on_batch, init=fn_carry_init, xs=args_batched)
        # re-merge scan batches
        y = pytree_combine_batches(y, ignore_last_n_elements_from_last_batch=ignore_n_from_last_batch)
        chex.assert_tree_shape_prefix(y, (num_elements,))
        if fn_has_carry:
            return carry, y
        else:
            return y


def scan_batched(
    fn: Callable[[Any, Any], tuple[Any, Any]],
    x: any,
    batch_size: int,
    fn_carry_init: Any = None,
    batch_axis_needs_to_be_dividable_by_batch_size=True
) -> tuple[Any, Any]:
    """
    Just as scan, apply fn to each of the input args (over the first batch axis).
    But if the batch axis size is larger than batch size, split args into batches and scan over them.
    Otherwise, directly call fn(*args)
    :param fn: function to apply to each element. Has signature fn(carry, x) -> carry, y
    :param x: the data to pass to fn, may be split into batches along first axis.
    :param batch_axis_needs_to_be_dividable_by_batch_size: if true,
                args batch axis size needs to be dividable by batch size if args batch axis size < batch_size.
                Otherwise, may throw some padded values from vmap away.
    :return: The result of fn of each batch element but all batches recombined (first axis is the batch axis).
    """
    num_elements = pytree_get_shape_first_axis_equal(x)
    # call directly
    if num_elements <= batch_size:
        carry, y = fn(fn_carry_init, x)
    else:
        # for too many elements we need to scan over batches
        args_batched, ignore_n_from_last_batch = pytree_split_in_batches_and_padd(
            x, batch_size=batch_size,
            fail_when_x_not_devidable_by_batch_size=batch_axis_needs_to_be_dividable_by_batch_size
        )
        carry, y = jax.lax.scan(fn, init=fn_carry_init, xs=args_batched)
        # re-merge scan batches
        y = pytree_combine_batches(y, ignore_last_n_elements_from_last_batch=ignore_n_from_last_batch)
        chex.assert_tree_shape_prefix(y, (num_elements,))
    return carry, y