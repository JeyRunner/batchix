from unittest import TestCase

import chex
import jax
import jax.numpy as jnp
import pytest

from batchix.batching import pytree_split_in_batches_with_remainder, pytree_combine_batches, pytree_sub_index_each_leaf
from batchix.tree_shape import pytree_get_shape_last_n_equal
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