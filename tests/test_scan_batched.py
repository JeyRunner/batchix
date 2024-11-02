import unittest
from unittest import TestCase

import chex
import jax.numpy as jnp
import pytest

from batchix.vmap_scan import scan_batched
from tests.common import make_test_pytree


class TestVmapScan(TestCase):

    def test_scan_batched_fit_None(self):
        data = make_test_pytree(30)

        def process_batch(carry, x):
            return carry, x['a']*2

        carry, out = scan_batched(
            process_batch,
            data,
            batch_size=10,
            batch_remainder_stategy='None'
        )
        chex.assert_trees_all_equal(out, data['a']*2)


    def test_scan_batched_ExtraLastBatch(self):
        data = make_test_pytree(35)

        def process_batch(carry, x):
            return carry, x['a']*2

        carry, out = scan_batched(
            process_batch,
            data,
            batch_size=10,
            batch_remainder_stategy='ExtraLastBatch'
        )
        chex.assert_trees_all_equal(out, data['a']*2)


    def test_scan_batched_fit_ExtraLastBatch(self):
        data = make_test_pytree(30)

        def process_batch(carry, x):
            return carry, x['a']*2

        carry, out = scan_batched(
            process_batch,
            data,
            batch_size=10,
            batch_remainder_stategy='ExtraLastBatch'
        )
        chex.assert_trees_all_equal(out, data['a']*2)


    def test_scan_batched_PadAndExtraLastBatch(self):
        data = make_test_pytree(35)

        def process_batch(carry, x, valid_x_mask, invalid_last_n_elements_in_x: int):
            y = x['a']*2
            y = y[valid_x_mask]
            sum = jnp.sum(x['a'][valid_x_mask])
            return carry + sum, y

        carry, out = scan_batched(
            process_batch,
            data,
            batch_size=10,
            fn_carry_init=0,
            batch_remainder_stategy='PadAndExtraLastBatch'
        )
        chex.assert_trees_all_equal(out, data['a']*2)
        chex.assert_trees_all_equal(carry, jnp.sum(data['a']))


