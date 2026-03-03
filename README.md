# CUDA Matrix Multiplication

CUDA implementations of matrix multiplication with progressive optimizations.

## Implementations

| File | Description |
|------|-------------|
| `matmul.cu` | Naive matrix multiplication |
| `matmul_tiled.cu` | Tiled (shared memory) matrix multiplication |
| `matmul_tiled_thread_coarsening.cu` | Tiled + thread coarsening |

## Benchmarks (NVIDIA B200, 8192x8192)

| Implementation | Kernel time (ms) |
|----------------|-----------------|
| `matmul` | 208.686 |
| `matmul_tiled` | 121.479 |
| `matmul_tiled_thread_coarsening` | 111.538 |

## Usage

```bash
# Compile and run (default: matmul)
make run

# Specify a target
make run TARGET=matmul_tiled
make run TARGET=matmul_tiled_thread_coarsening

# Profile with Nsight Systems
make profile TARGET=matmul_tiled
```

## Requirements

- CUDA toolkit with `nvcc`
- NVIDIA GPU
