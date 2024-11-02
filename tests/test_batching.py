from unittest import TestCase

import chex
import jax.numpy as jnp
import pytest

from batchix.batching import pytree_split_in_batches_with_remainder, pytree_combine_batches, pytree_sub_index_each_leaf
from tests.common import make_test_pytree


class TestBatching(TestCase):
    def test_pytree_split_in_batches_with_remainder_fail(self):
        data = make_test_pytree(15)
        with pytest.raises(Exception):
            data_batched, data_batched_remainder = pytree_split_in_batches_with_remainder(
                data,
                batch_size=10,
                batch_remainder_strategy='None'
            )

    def test_pytree_split_in_batches_with_remainder_batch_fits(self):
        data = make_test_pytree(20)
        data_batched, data_batched_remainder = pytree_split_in_batches_with_remainder(
            data,
            batch_size=10,
            batch_remainder_strategy='None'
        )
        self.assertEqual(data_batched_remainder, None)
        chex.assert_tree_shape_prefix(data_batched, (2, 10))
        data_recombined = pytree_combine_batches(data_batched)
        chex.assert_trees_all_equal(data, data_recombined)


    def test_pytree_split_in_batches_with_remainder_batch_Pad(self):
        data = make_test_pytree(35)
        data_batched, data_batched_remainder = pytree_split_in_batches_with_remainder(
            data,
            batch_size=10,
            batch_remainder_strategy='Pad'
        )
        self.assertEqual(data_batched_remainder, 5)
        chex.assert_tree_shape_prefix(data_batched, (4, 10))
        data_recombined = pytree_combine_batches(data_batched, data_batched_remainder)
        chex.assert_trees_all_equal(data, data_recombined)


    def test_pytree_split_in_batches_with_remainder_batch_ExtraLastBatch(self):
        data = make_test_pytree(35)
        data_batched, data_batched_remainder = pytree_split_in_batches_with_remainder(
            data,
            batch_size=10,
            batch_remainder_strategy='ExtraLastBatch'
        )
        chex.assert_trees_all_equal(data_batched_remainder, pytree_sub_index_each_leaf(data, jnp.s_[-5:]))
        chex.assert_tree_shape_prefix(data_batched, (3, 10))
        data_recombined = pytree_combine_batches(data_batched, data_batched_remainder)
        chex.assert_trees_all_equal(data, data_recombined)

    def test_pytree_split_in_batches_with_remainder_batch_fit__ExtraLastBatch(self):
        data = make_test_pytree(40)
        data_batched, data_batched_remainder = pytree_split_in_batches_with_remainder(
            data,
            batch_size=10,
            batch_remainder_strategy='ExtraLastBatch'
        )
        chex.assert_tree_shape_prefix(data_batched, (4, 10))
        data_recombined = pytree_combine_batches(data_batched, data_batched_remainder)
        chex.assert_trees_all_equal(data, data_recombined)