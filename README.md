# microbenchies

A collection of GPU microbenchmarks for understanding performance characteristics of common deep learning operations.

## Index

| Folder | Description |
|--------|-------------|
| [fused_matmul](./fused_matmul/) | Tests whether fusing two matmuls `X @ W1` and `X @ W2` into a single `X @ [W1\|W2]` is faster. Findings: fusing is generally 5-15% faster due to reduced memory reads of X, with sweet spot at batch sizes 32-512. |
