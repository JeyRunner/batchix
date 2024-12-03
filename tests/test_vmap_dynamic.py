from unittest import TestCase

import chex

from batchix.vmap_scan_dynamic import DynamicVMapFlexibleLength, scan_batched_dynamic_padded
from tests.common import make_test_pytree


class TestVmapDynamic(TestCase):

    def test_DynamicVMapFlexibleLength(self):
        data = make_test_pytree(35)

        def process_batch(x):
            return x['a']*2

        vmap = DynamicVMapFlexibleLength(
            vmap_batch_size=10,
            fn=process_batch
        )
        out = vmap(data)
        chex.assert_trees_all_equal(out, data['a']*2)


    def test_scan_batched_dynamic(self):
        data = make_test_pytree(35)

        #@jax.vmap
        def process_batch(carry, x):
            print(carry)
            carry += 1
            return carry, x['a']*2

        carry, out = scan_batched_dynamic_padded(
            process_batch,
            x=data,
            batch_size=10,
            fn_carry_init=0
        )
        chex.assert_equal(carry, 4)
        chex.assert_trees_all_equal(out, data['a']*2)



