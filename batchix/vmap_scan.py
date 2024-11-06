from typing import TypeVar, Callable, Any, Literal, Protocol

import chex
import jax
from jax import numpy as jnp

from batchix.batching import pytree_split_in_batches_with_remainder, pytree_combine_batches, pytree_pad, \
    pytree_sub_index_each_leaf
from batchix.tree_shape import (
    pytree_get_shape_first_axis_equal,
)

X = TypeVar("X")
Y = TypeVar("Y")


def vmap_batched_scan(
    fn: Callable[[...], Any] | Callable[[Any, ...], tuple[Any, Any]],
    args: tuple,
    batch_size: int,
    fn_has_carry: bool = False,
    fn_carry_init: Any = None,
    batch_axis_needs_to_be_dividable_by_batch_size=True,
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
            carry, y = jax.vmap(fn, in_axes=(None,) + (0,) * len(args))(carry, *x)
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
        args_batched, batch_remainder = pytree_split_in_batches_with_remainder(
            args,
            batch_size=batch_size,
            batch_remainder_strategy='None' if batch_axis_needs_to_be_dividable_by_batch_size else 'Pad'
        )
        carry, y = jax.lax.scan(vmap_on_batch, init=fn_carry_init, xs=args_batched)
        # re-merge scan batches
        y = pytree_combine_batches(
            y, batch_remainder=batch_remainder
        )
        chex.assert_tree_shape_prefix(y, (num_elements,))
        if fn_has_carry:
            return carry, y
        else:
            return y


Carry = TypeVar('Carry')
Z = TypeVar('Z')
class ScanBatchedFn(Protocol):
    def __call__(self, carry: Carry, x: X) -> tuple[Carry, Y] | tuple[Carry, Y, Z]:
        """
        Process batch x and return new carry state as well as output y.
        Optionally also return a third value that collapse/summarizes the batch samples (does not have first batch dim).
        """

class ScanBatchedExtraLastBatchFn(Protocol):
    def __call__(self, carry: Carry, x: X, valid_x_mask, invalid_last_n_elements_in_x: int = 0) -> tuple[Carry, Y] | tuple[Carry, Y, Z]:
        """
        Process batch x and return new carry state as well as output y.
        Optionally also return a third value that collapse/summarizes the batch samples (does not have first batch dim).
        When called on the last batch the `invalid_last_n_elements_in_x` last elements in x will be padded/invalid.
        These should not be used for computations changing the returned carry and computing the returned Y.
        Thus if `invalid_last_n_elements_in_x` =! 0 ensure
        that the output of this function depends not on the values x[valid_x_mask].
        E.g.
            return carry, (x*2)[valid_x_mask]
        """


def scan_batched(
    fn: ScanBatchedFn | ScanBatchedExtraLastBatchFn,
    x: X,
    batch_size: int,
    fn_carry_init: Carry = None,
    batch_remainder_strategy: Literal['None', 'PadValidSampled', 'ExtraLastBatch', 'PadAndExtraLastBatch'] = 'None',
    batch_remainder_call_change_carry: bool = True,
    rng_key_sample_padding: jax.random.PRNGKey = None
) -> tuple[Carry, Y] | tuple[Carry, Y, Z]:
    """
    Just as scan, apply fn to each of the input args (over the first batch axis).
    If the batch axis size is larger than batch size, split args into batches and scan over them.
    Otherwise, directly call fn(*args)
    Note that the output first axis size of returned y will be equal to given x.
    :param fn: function to apply to each element. Returns: carry, y.
                Note that if this function does not return None for y, y needs to have same shape as valid x elements.
                Optionally also returns a third value that collapse/summarizes the batch samples (does not have first batch dim).
    :param x: the data to pass to fn, may be split into batches along first axis.
    :param batch_remainder_strategy:
        How to handle the case if the number of elements in x is not dividable by batch_size:
        'None': fail if there is a remainder.
        'PadValidSampled': Pad missing elements in last batch with random samples from x,
                            Thus also these elements are valid and second return value is None.
                            In this case rng_key_sample_padding has to be provided.
                            For example this can be used for training a model and the train data is randomized anyway.
        'ExtraLastBatch': Do an extra last call to fn for last smaller batch, then final fn call gets a smaller x.
        'PadAndExtraLastBatch': Do an extra last call to fn for last batch which is padded to the batch_size,
                                thus the last n elements of given x are invalid.
                                Here ScanBatchedExtraLastBatchFn has to be used for fn.
    :param rng_key_sample_padding: Sampling key for batch_remainder_strategy=PadValidSampled.
    :return: The result of fn of each batch element but all batches recombined (first axis is the batch axis).
                Optionally returns a third value when fn also returns a third value.
    """
    num_elements = pytree_get_shape_first_axis_equal(x)

    def scan_fn(carry: Carry, x: X, invalid_last_n_elements_in_x: int = 0) -> tuple[Carry, Y]:
        if batch_remainder_strategy == 'PadAndExtraLastBatch':
            n_invalid = None if invalid_last_n_elements_in_x == 0 else invalid_last_n_elements_in_x
            o = fn(
                carry, x,
                valid_x_mask=jnp.s_[:-n_invalid if n_invalid is not None else None],
                invalid_last_n_elements_in_x=n_invalid
            )
        else:
            o = fn(carry, x)
        if len(o) > 2:
            carry, y, summary = o
            return carry, (y, summary)
        else:
            return o

    summary_per_batch = None

    # call directly
    if num_elements <= batch_size:
        carry, y = scan_fn(fn_carry_init, x)
        if isinstance(y, tuple):
            y, summary = y
    else:
        # for too many elements we need to scan over batches
        x_batched, batch_remainder = pytree_split_in_batches_with_remainder(x,
            batch_size=batch_size,
            batch_remainder_strategy=(
                'ExtraLastBatch' if batch_remainder_strategy == 'PadAndExtraLastBatch' else batch_remainder_strategy
            ),
            rng_key=rng_key_sample_padding
        )
        carry, y = jax.lax.scan(scan_fn, init=fn_carry_init, xs=x_batched)

        # extra call for ExtraLastBatch
        if batch_remainder_strategy.endswith('ExtraLastBatch') and batch_remainder is not None:
            if batch_remainder_strategy == 'PadAndExtraLastBatch':
                batch_remainder, invalid_n_last_in_last_batch = pytree_pad(batch_remainder, pad_to_size=batch_size)
                carry_new, batch_remainder = scan_fn(carry, batch_remainder, invalid_last_n_elements_in_x=invalid_n_last_in_last_batch)
            elif batch_remainder_strategy == 'ExtraLastBatch':
                carry_new, batch_remainder = scan_fn(carry, batch_remainder)
            if batch_remainder_call_change_carry:
                carry = carry_new

        # check if y is tuple and contains a second summary element per batch
        if isinstance(y, tuple):
            y, summary = y
            batch_remainder, batch_remainder_summary = batch_remainder
            summary, batch_remainder_summary = pytree_sub_index_each_leaf((summary, batch_remainder_summary), jnp.s_[jnp.newaxis])
            summary_per_batch = pytree_combine_batches(
                summary, batch_remainder=batch_remainder_summary
            )

        # re-merge scan batches
        y = pytree_combine_batches(
            y, batch_remainder=batch_remainder
        )
        chex.assert_tree_shape_prefix(y, (num_elements,))

    if summary_per_batch is None:
        return carry, y
    else:
        return carry, y, summary_per_batch

