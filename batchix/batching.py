import math
from typing import TypeVar, Callable, Tuple, Any

import chex
import jax
from einshape import jax_einshape as einshape
from jax import numpy as jnp
from jaxtyping import PyTree, Shaped, Array, Integer
from numpy.lib._index_tricks_impl import IndexExpression

from batchix.tree_shape import pytree_get_shape_first_axis_equal, pytree_get_shape_first_n_equal

X = TypeVar('X')
Y = TypeVar('Y')





def pytree_split_in_batches_and_padd[T](x: T, batch_size, fail_when_x_not_devidable_by_batch_size=False) -> tuple[T, int]:
    """
    Splits pytree x into batches by adding a first batch dimension.
    The elements of the last batch will be padded so that all batches have the same size
    :param x:
    :param batch_size:
    :return: (x split into batches, x_remain_elements that need to be ignored from the last batch)
    """

    # fill x margin with last data element
    x_shape_zero: int = pytree_get_shape_first_n_equal(x, first_n_shape_elements=1)[0]
    num_batches = int(math.ceil(x_shape_zero / batch_size))
    x_remain_elements = num_batches * batch_size - x_shape_zero

    if x_remain_elements != 0 and not fail_when_x_not_devidable_by_batch_size:
        x = jax.tree_util.tree_map(
            lambda el: jnp.concat([el, jnp.repeat(el[-2:-1], axis=0, repeats=x_remain_elements)], axis=0),
            x
        )
    else:
        assert x_remain_elements == 0, \
            f'x first dim needs to be dividable by batch size, but x.shape[0]={x_shape_zero} and batch size is {batch_size}'

    # split into batches
    x_batched = jax.tree_util.tree_map(lambda x: jnp.stack(jnp.split(x, axis=0, indices_or_sections=num_batches), axis=0), x)

    chex.assert_tree_shape_prefix(x_batched, (num_batches, batch_size))
    return x_batched, x_remain_elements


def pytree_split_in_batches[T](x: T, batch_size, num_batches) -> T:
    """
    Splits pytree x into batches by adding a first batch dimension.
    :param x:
    :param batch_size:
    :return: x split into batches
    """
    chex.assert_tree_shape_prefix(x, (batch_size*num_batches,))
    x_batched = jax.tree_util.tree_map(
        lambda x: jnp.stack(jnp.split(x, axis=0, indices_or_sections=num_batches), axis=0), x)
    return x_batched


def pytree_combine_batches(
    x: PyTree[Shaped[Array, 'num_batches batch_size ...'], 'T'],
    ignore_last_n_elements_from_last_batch=0
) -> PyTree[Shaped[Array, 'num_batches*batch_size ...'], 'T']:
    """
    Stacks the first batch axis elements. This removes the first batch axis.
    Assumes that the input pytree elements have the shape (num_batches, batch_size, ...)
    :param x: the input pytree
    :param ignore_last_n_elements_from_last_batch: remove these last elements form all leafs of pytree (after removing first axis)
    :return: pytree with flatted batches and removed first dim.
    """
    num_batches, batch_size = pytree_get_shape_first_n_equal(x, first_n_shape_elements=2)
    out = jax.tree_util.tree_map(lambda x: einshape('ab...->(ab)...', x)[:-ignore_last_n_elements_from_last_batch or None], x)
    chex.assert_tree_shape_prefix(out, (num_batches*batch_size - ignore_last_n_elements_from_last_batch, ))
    return out


def pytree_concatenate_each_leaf(
    x: list[PyTree['T']],
    axis: int = 0
) -> PyTree['T']:
    """
    Concatenates each leaf with its corresponding leafs in the pytree list x.
    :param x:
    """
    return jax.tree_util.tree_map(lambda *leafs: jnp.concatenate(leafs, axis=axis), *x)


def pytree_sub_index_each_leaf(
    x: PyTree['T'],
    index: IndexExpression | Integer[Array, '...']
) -> PyTree['T']:
    """
    Takes just the elements of the index expression of each leaf in the tree.
    :param x:
    """
    return jax.tree_util.tree_map(lambda l: l[index], x)


def pytree_squeeze_first_axis(x: PyTree['T']):
    """
    Removes fist axis from all pytree leafs, the first axis has to be size 1.
    """
    assert pytree_get_shape_first_axis_equal(x) == 1, "First axis needs to be of size 1"
    return jax.tree_util.tree_map(lambda l: l[0], x)
