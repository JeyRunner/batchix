import math
from argparse import ArgumentError
from typing import TypeVar, Literal

import chex
import jax
import numpy as np
from einshape import jax_einshape as einshape
from jax import numpy as jnp
from jaxtyping import PyTree, Shaped, Array, Integer
from numpy.lib._index_tricks_impl import IndexExpression

from batchix.tree_shape import (
    pytree_get_shape_first_axis_equal,
    pytree_get_shape_first_n_equal,
)

X = TypeVar("X")
Y = TypeVar("Y")


def pytree_pad(
    x: X, pad_to_size: int = None, pad_add_elements: int = None
) -> tuple[X, int]:
    """
    Pads pytree along the first axis by repeating the last element in the first axis.
    :param x: input pytree to pad
    :param pad_to_size: the returned pytree will have the first axis with this size
    :param pad_add_elements: add this number of elements to the first axis. Can just be used when pad_to_size is not given.
    :return: padded pytree, pad_add_elements
    """
    assert (pad_to_size is not None) ^ (pad_add_elements is not None), "either provide pad_to_size or pad_add_elements"
    x_shape_zero = pytree_get_shape_first_axis_equal(x)
    if pad_add_elements is None:
        pad_add_elements = pad_to_size - x_shape_zero
    assert pad_add_elements >= 0, f"can't pad to size ({pad_to_size}) smaller than input size {x_shape_zero}"
    x = jax.tree_util.tree_map(
        lambda el: jnp.concat([el, jnp.repeat(el[-2:-1], axis=0, repeats=pad_add_elements)], axis=0),
        x
    )
    return x, pad_add_elements

def pytree_split_in_batches_with_remainder(
    x: X, batch_size: int,
    batch_remainder_strategy: Literal['None', 'Pad', 'ExtraLastBatch'] = 'None'
) -> tuple[X, int | X | None]:
    """
    Splits pytree x into batches by adding a first batch dimension.
    If number of elements in the first dimension of x is dividable by num_batches,
    the elements of the last batch will be padded so that all batches have the same size
    or the last batch with a smaller size is returned separately, see `batch_remainder_strategy`.
    :param x: the pytree to split into batches
    :param batch_size: number of elements in each batch
    :param batch_remainder_strategy:
        How to handle the case if the number of elements in x is not dividable by batch_size:
        'None': fail if there is a remainder.
        'Pad': Pad missing elements in last batch, return the number of padded elements as second value.
        'ExtraLastBatch': Return the last batch with a smaller size as second value.
    :return: (
        x split into batches,
        x_remain_elements that need to be ignored from the last batch or last batch with smaller size
        [depending on batch_remainder_strategy]
        )
    """

    # fill x margin with last data element
    x_shape_zero: int = pytree_get_shape_first_n_equal(x, first_n_shape_elements=1)[0]
    num_batches = int(math.ceil(x_shape_zero / batch_size))
    x_remain_elements = num_batches * batch_size - x_shape_zero

    batch_remainder: int | X = None
    if x_remain_elements != 0:
        if batch_remainder_strategy == 'None':
            assert x_remain_elements == 0, (
                f"x first dim needs to be dividable by batch size, but x.shape[0]={x_shape_zero} and batch size is {batch_size}. \n"
                "Consider setting batch_remainder_strategy to 'Pad' or 'ExtraLastBatch'."
            )
        elif batch_remainder_strategy == 'Pad':
            x, _ = pytree_pad(x, pad_add_elements=x_remain_elements)
            batch_remainder = x_remain_elements
        elif batch_remainder_strategy == 'ExtraLastBatch':
            batch_remainder = pytree_sub_index_each_leaf(x, jnp.s_[-x_remain_elements:])
            x = pytree_sub_index_each_leaf(x, jnp.s_[:-x_remain_elements])
            num_batches = num_batches - 1
        else:
            raise ArgumentError(batch_remainder_strategy, '')

    # split into batches
    x_batched = pytree_split_in_batches(x, batch_size=batch_size, num_batches=num_batches)

    chex.assert_tree_shape_prefix(x_batched, (num_batches, batch_size))
    return x_batched, batch_remainder


def pytree_split_in_batches(x: X, batch_size: int, num_batches: int | None = None) -> X:
    """
    Splits pytree x into batches by adding a first batch dimension.
    The number of elements in the first dimension of x has to be dividable by num_batches.
    :param x: the pytree to split into batches
    :param batch_size: number of elements in each batch
    :return: x split into batches
    """
    x_shape_zero = pytree_get_shape_first_axis_equal(x)
    assert x_shape_zero % batch_size == 0, (
        "number of elements in x needs to be devidable by batch_size. "
        "Consider using pytree_split_in_batches_with_remainder"
    )
    num_batches_ = int(x_shape_zero / batch_size)
    assert (num_batches is None) or num_batches == num_batches_
    num_batches = num_batches_

    chex.assert_tree_shape_prefix(x, (batch_size * num_batches,))
    x_batched = jax.tree_util.tree_map(
        lambda x: jnp.stack(jnp.split(x, axis=0, indices_or_sections=num_batches), axis=0),
        x
    )
    return x_batched


def pytree_combine_batches(
    x: PyTree[Shaped[Array, "num_batches batch_size ..."], "T"],
    batch_remainder: int | X | None = None,
) -> PyTree[Shaped[Array, "num_batches*batch_size ..."], "T"]:
    """
    Stacks the first batch axis elements. This removes the first batch axis.
    Assumes that the input pytree elements have the shape (num_batches, batch_size, ...).
    If batched x was produced via pytree_split_in_batches_with_remainder, the batch_remainder argument needs to be provided.
    :param x: the batched pytree
    :param batch_remainder: either the number of padded elements in the last batch or the last batch with size<batch_size.
    :return: pytree with flatted batches and removed first dim.
    """

    num_batches, batch_size = pytree_get_shape_first_n_equal(x, first_n_shape_elements=2)
    out_sub_idx = jnp.s_[:]
    final_x_shape_offset = 0
    last_batch = None
    if np.isscalar(batch_remainder):
        out_sub_idx = jnp.s_[:-batch_remainder]
        final_x_shape_offset = -batch_remainder
    elif batch_remainder is not None:
        chex.assert_trees_all_equal_structs(x, batch_remainder)
        last_batch = batch_remainder
        final_x_shape_offset = pytree_get_shape_first_axis_equal(batch_remainder)

    out = jax.tree_util.tree_map(
        lambda el: einshape('ab...->(ab)...', el)[out_sub_idx],
        x
    )
    if last_batch is not None:
        out = jax.tree_util.tree_map(lambda x, last: jnp.concat([x, last], axis=0), out, last_batch)

    chex.assert_tree_shape_prefix(out, (num_batches*batch_size + final_x_shape_offset, ))
    return out


def pytree_concatenate_each_leaf(x: list[PyTree["T"]], axis: int = 0) -> PyTree["T"]:
    """
    Concatenates each leaf with its corresponding leafs in the pytree list x.
    :param x:
    """
    return jax.tree_util.tree_map(lambda *leafs: jnp.concatenate(leafs, axis=axis), *x)


def pytree_sub_index_each_leaf(
    x: PyTree["T"], index: IndexExpression | Integer[Array, "..."] | int
) -> PyTree["T"]:
    """
    Takes just the elements of the index expression of each leaf in the tree.
    :param x:
    """
    return jax.tree_util.tree_map(lambda l: l[index], x)


def pytree_squeeze_first_axis(x: PyTree):
    """
    Removes fist axis from all pytree leafs, the first axis has to be size 1.
    """
    assert pytree_get_shape_first_axis_equal(x) == 1, "First axis needs to be of size 1"
    return jax.tree_util.tree_map(lambda l: l[0], x)