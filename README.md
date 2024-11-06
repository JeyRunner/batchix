# Batchix
This is utility package for jax for handing batched data.
Its features include:
 - splitting pytrees in to batches of fixed size 
 - recombining a pytree split into batches
 - batched scan (scan over batches)
 - batched vmap (scan over vmap)
 - automatic handling of cases where the data to be split into batches is not dividable by the batch size
    - either pad or make a separate last smaller batch

## Install
```bash
pip install git+https://github.com/JeyRunner/batchix.git
```

## Usage
### Split and Merge Batches
Split array into batches:

```python
data_batched, data_batched_remainder = pytree_split_in_batches_with_remainder(
    x=jnp.arange(15),
    batch_size=10,
    # takes care of the data not being dividable by the batch size
    # data_batched_remainder is last batch that has smaller batch size
    batch_remainder_strategy='ExtraLastBatch'
)

# transform data_batched, data_batched_remainder
# e.g vmap/scan over data_batched and direct processing of last single batch data_batched_remainder

data_recombined = pytree_combine_batches(data_batched, data_batched_remainder)
```

### Scan over Batches
Split pytree in batches and scan over the batches:
```python
def process_batch(carry, x):
    # for last batch x has shape (5,) otherwise (10,)
    return carry, x*2

carry, out = scan_batched(
    process_batch,
    x=jnp.arange(15),
    batch_size=10,
    # takes care of the data not being dividable by the batch size
    # makes separate call to process_batch for the last remaining elements
    batch_remainder_stategy='ExtraLastBatch'
)
```

Split pytree in batches and scan over the batches with manual padding handling in scan body:
```python
def process_batch(carry, x, valid_x_mask, invalid_last_n_elements_in_x):
    # x has allways shape (10,)
    # but for the last batch, last n elements or x are invalid padded values
    y = x*2
    y = y[valid_x_mask] # important: just use valid x elements
    return carry, y

carry, out = scan_batched(
    process_batch,
    x=jnp.arange(15),
    batch_size=10,
    # takes care of the data not being dividable by the batch size
    # makes separate call to process_batch for the last remaining elements
    batch_remainder_stategy='PadAndExtraLastBatch'
)
```
When the data size is dividable by the batch size both of the above example will also work fine.


### Dev setup
Install deps:
```bash
pip install .[dev, test]
```