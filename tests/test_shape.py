from unittest import TestCase

import chex
import jax
import jax.numpy as jnp
import pytest

from batchix.batching import pytree_split_in_batches_with_remainder, pytree_combine_batches, pytree_sub_index_each_leaf, \
    pytree_dynamic_slice_in_dim
from batchix.tree_shape import pytree_get_shape_last_n_equal, pytree_get_shape_at_axis_equal
from tests.common import make_test_pytree


class TestBatching(TestCase):
    def test_pytree_get_shape_last_n_equal(self):
        data = dict(
            a=jnp.ones((10, 3, 5)),
            b=jnp.ones((10, 90, 5)),
        )
        with pytest.raises(Exception):
            pytree_get_shape_last_n_equal(data, last_n_shape_elements=2)

        shape_last = pytree_get_shape_last_n_equal(data, last_n_shape_elements=1)
        self.assertEqual(shape_last, (5,))
        self.assertEqual(pytree_get_shape_at_axis_equal(data, axis=0), 10)
        self.assertEqual(pytree_get_shape_at_axis_equal(data, axis=-1), 5)
        with self.assertRaises(AssertionError):
            pytree_get_shape_at_axis_equal(data, axis=1)



    def test_pytree_dynamic_slice_in_dim(self):
        data = dict(
            a=jnp.arange(100).reshape(-1,  5),
            b=jnp.arange(10).reshape(-1,  5)
        )

        slice_invalid = pytree_dynamic_slice_in_dim(data, start_index=2, slice_size=4, axis=-1, if_out_of_bounds_set_all_nan=True)
        self.assertTrue(jnp.all(jnp.isnan(slice_invalid['a'])))

        slice_valid = pytree_dynamic_slice_in_dim(data, start_index=1, slice_size=4, axis=-1, if_out_of_bounds_set_all_nan=True)
        chex.assert_trees_all_equal(
            slice_valid,
            pytree_sub_index_each_leaf(data, jnp.s_[..., 1:1+4])
        )
