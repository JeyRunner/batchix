from unittest import TestCase

import chex
import jax
import jax.numpy as jnp
import pytest

from batchix.batching import pytree_split_in_batches_with_remainder, pytree_combine_batches, pytree_sub_index_each_leaf, \
    pytree_dynamic_slice_in_dim, pytree_dynamic_slice_in_dim__oob_set_nan, pytree_dynamic_slice_in_dim__vmapped, \
    pytree_dynamic_slice_in_dim__vmapped__oob_set_nan
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


    def test_pytree_dynamic_slice_in_dim__vmapped(self):
        data = dict(
            a=jnp.arange(50).reshape(5, -1,  10),
            b=jnp.arange(100).reshape(5, -1,  10)
        )

        start_indies = jnp.arange(5)

        slice_valid = pytree_dynamic_slice_in_dim__vmapped__oob_set_nan(data, vmap_axis=0, axis=-1, slice_size=4, start_indies=start_indies)
        for i in range(start_indies.shape[0]):
            start_idx = start_indies[i]
            chex.assert_trees_all_equal(
                pytree_sub_index_each_leaf(slice_valid, jnp.s_[i, :, :]),
                pytree_sub_index_each_leaf(data, jnp.s_[i, :, start_idx:start_idx+4])
            )

        slice_invalid = pytree_dynamic_slice_in_dim__vmapped__oob_set_nan(data, vmap_axis=0, axis=-1, slice_size=7, start_indies=start_indies)
        # last has nans (index 4...(4+7) = 4...11 = elemets 4...10)
        self.assertFalse(jnp.any(jnp.isnan(slice_invalid['a'][-2])), f"{slice_invalid['a'][-2]}")
        self.assertTrue(jnp.all(jnp.isnan(slice_invalid['a'][-1])), f"{slice_invalid['a'][-1]}")
