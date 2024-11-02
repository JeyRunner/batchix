import unittest
from unittest import TestCase

import chex
import jax.numpy as jnp
import pytest

from batchix.vmap_scan import scan_batched, vmap_batched_scan
from tests.common import make_test_pytree


class TestVmapScan(TestCase):

    def test_scan_batched_fit_None(self):
        data = make_test_pytree(30)

        def process_el(carry, x):
            return carry, x['a']*2

        carry, out = vmap_batched_scan(
            process_el,
            data,
            batch_size=10,
            batch_axis_needs_to_be_dividable_by_batch_size=False,
            fn_has_carry=True
        )
        chex.assert_trees_all_equal(out, data['a']*2)

