from typing import Tuple

import jax
from jaxtyping import PyTree


def pytree_get_shape_first_n_equal(
    x: PyTree, first_n_shape_elements=1, custom_error_msg: str = None
) -> Tuple[int, ...]:
    """
    Get the first n elements of the shape of the leafs of the given pytree x.
    This assumes/checks that the first n elements of all the leafs of the pytree have the same shape.
    :param x: the pytree
    :param first_n_shape_elements: return the first n elements of the shape of the pytree leafs
    :return: return the first n elements of the shape of the pytree leafs
    """
    if custom_error_msg is not None:
        custom_error_msg = custom_error_msg + ": "
    else:
        custom_error_msg = ''

    def reduce_x_shape_zero(pre_shape, el):
        assert (
            len(el.shape) >= first_n_shape_elements
        ), f"{custom_error_msg}Tree element {el} (with shape {el.shape}) does not have the required number of {first_n_shape_elements} axis."
        if pre_shape is not None:
            assert el.shape[:first_n_shape_elements] == pre_shape, (
                f"{custom_error_msg}all pytree leafs need to have the first {first_n_shape_elements} shape sizes. "
                f"Tree element {el} (with shape {el.shape[:first_n_shape_elements]}) does not have the same shape as previous elements {pre_shape}"
            )

        return el.shape[:first_n_shape_elements]

    shape_prefix = jax.tree_util.tree_reduce(reduce_x_shape_zero, x, initializer=None)
    return shape_prefix


def pytree_get_shape_first_axis_equal(x: PyTree) -> int:
    """
    Get the first element of the shape of the leafs of the given pytree x.
    This assumes/checks that the first element/axis of all the leafs of the pytree have the same shape.
    :param x: the pytree
    :return: return the size of the first axis of all leafs.
    """
    return pytree_get_shape_first_n_equal(x, first_n_shape_elements=1)[0]


def pytree_get_shape_last_n_equal(
    x: PyTree, last_n_shape_elements=1
) -> Tuple[int, ...]:
    """
    Get the last n elements of the shape of the leafs of the given pytree x.
    This assumes/checks that the last n elements of all the leafs of the pytree have the same shape.
    :param x: the pytree
    :param last_n_shape_elements: return the last n elements of the shape of the pytree leafs.
    :return: return the last n elements of the shape of the pytree leafs,
    """

    def reduce_x_shape_zero(pre_shape, el):
        if pre_shape is not None:
            assert el.shape[-last_n_shape_elements:] == pre_shape, (
                f"all pytree leafs need to have the last {last_n_shape_elements} shape sizes. "
                f"Tree element {el} does not have the same shape as previous elements {pre_shape}"
            )
        return el.shape[-last_n_shape_elements:]

    shape_suffix = jax.tree_util.tree_reduce(reduce_x_shape_zero, x, initializer=None)
    return shape_suffix


def pytree_get_shape_last_axis_equal(
    x: PyTree,
) -> int:
    """
    Get the last element of the shape of the leafs of the given pytree x.
    This assumes/checks that the last element of all the leafs of the pytree have the same shape.
    :param x: the pytree
    :return: return the last element of the shape of the pytree leafs
    """
    return pytree_get_shape_last_n_equal(x, last_n_shape_elements=1)[0]
