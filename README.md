# Batchix
This is a batched data utility package for jax.
Its features include:
 - splitting pytrees in to batches of fixed size 
 - recombining a pytree split into batches
 - batched scan (scan over batches)
 - batched vmap (scan over vmap)


### Dev setup
Install deps:
```bash
pip install .[dev, test]
```