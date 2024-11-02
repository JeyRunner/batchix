from typing import TypeVar, Callable, Tuple, Any

import chex
import jax
from einshape import jax_einshape as einshape
from jax import numpy as jnp
from jaxtyping import PyTree, Shaped, Array, Integer
from numpy.lib.index_tricks import IndexExpression

X = TypeVar('X')
Y = TypeVar('Y')

class JaxVMapFlexibleLength:
  """
  Executed fn parallel of different batches of the input data input
  :param fn:
  :param vmap_batch_size:
  """

  def __init__(self, fn: Callable[[X], Y], vmap_batch_size: int):
    self.fn = fn
    self.vmap_batch_size = vmap_batch_size
    self.fn_jit = jax.jit(self.fn)
    self.vmap_fn_jitted = None

  def __call__(
      self,
      x: X
  ) -> Y:
    """
    Executed fn parallel of different batches of the input data input
    :param x: the input data (the first axis it the batch axis), the size of the first axis is flexible
    :return: y result of function batched in first axis
    """

    # fill x margin with last data element
    def reduce_x_shape_zero(pre_shape_zero, el):
      if pre_shape_zero is not None:
        assert el.shape[0] == pre_shape_zero
      return el.shape[0]

    x_shape_zero: int = jax.tree_util.tree_reduce(reduce_x_shape_zero, x, initializer=None)
    num_iterations = int(math.ceil(x_shape_zero / self.vmap_batch_size))
    x_remain_elements = num_iterations * self.vmap_batch_size - x_shape_zero

    x = jax.tree_util.tree_map(
      lambda el: jnp.concat([el, jnp.repeat(el[-2:-1], axis=0, repeats=x_remain_elements)], axis=0),
      x
    )

    if self.vmap_batch_size > 1:
      # manually cache the jitted function
      if self.vmap_fn_jitted == None:
        self.vmap_fn_jitted = jax.jit(jax.vmap(
          self.fn_jit
        ))

    result_all = None
    for i in range(num_iterations):
      x_sub = jax.tree_util.tree_map(
        lambda el: el[i * self.vmap_batch_size:(i + 1) * self.vmap_batch_size] if self.vmap_batch_size > 1 else el[
          i * self.vmap_batch_size], x)
      # test
      # print('shapes for input: ')
      # for el in x_sub:
      #     print('  - ', el.shape, ' dtype: ', el.dtype)

      # caching seems not to work for vmap
      if self.vmap_batch_size > 1:
        # print('-- vmapped fn jit cache size:', self.vmap_fn_jitted._cache_size())
        result = self.vmap_fn_jitted(x_sub)
      else:
        result = jax.jit(self.fn)(x_sub)

      if result_all is None:
        # create result tree with right sizes
        result_all = jax.tree_util.tree_map(lambda el: jnp.zeros(
          (num_iterations * self.vmap_batch_size,) + el.shape[1 if self.vmap_batch_size > 1 else 0:]), result)
      # set result
      # rich.print(result)
      # rich.print(result_all)
      result_all = jax.tree_util.tree_map(
        lambda el, new_el: (
          el.at[i * self.vmap_batch_size:(i + 1) * self.vmap_batch_size] if self.vmap_batch_size > 1 else el.at[
            i * self.vmap_batch_size]).set(new_el),
        result_all,
        result)
    # throw away margin
    return jax.tree_util.tree_map(lambda y: y[:x_shape_zero], result_all)