# Celeritas

Celeritas is a C++ large language model (LLM) inference framework.

It is designed around a clear separation of:

- `baseutil`: foundational runtime utilities (memory pool, tensor, CUDA helpers)
- `celeritas`: model-side operators and kernels
- `test`: unit and integration tests

## Goals

- Build an efficient and extensible C++ inference runtime
- Support heterogeneous backends (CPU/CUDA today, more backends later)
- Keep kernel dispatch and tensor/memory abstractions modular

## Repository Layout

```text
baseutil/
  memory/           # memory pools and memory manager
  tensor/           # tensor abstraction
  cudaUtil/         # CUDA utility wrappers
  CMakeLists.txt

celeritas/
  kernels/          # kernel implementations + kernel factory
  operation/        # higher-level ops (ongoing)
  CMakeLists.txt

test/
  testUtil/         # utility-level tests (pool/tensor)
  testOperation/    # op/kernel-level tests
  CMakeLists.txt
```

## Build Requirements

- CMake >= 3.18
- C++17 compiler
- CUDA Toolkit
- Armadillo
- GTest
- glog

## Build And Test

From repository root:

```bash
cmake -S test -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Current Status

- Tensor supports multi-dimensional shape metadata (`dims`)
- Tensor supports CPU/CUDA device conversion
- Tensor currently uses shared ownership semantics via internal ref-counting
- Kernel dispatch is handled by `KernelFactory`
- Example kernels: `add`, `emb` (CPU path available, CUDA path partially integrated)

## Roadmap

- More LLM kernels (matmul/rmsnorm/rope/mha/softmax)
- Unified backend registration mechanism (CPU/CUDA/OpenCL...)
- Graph-level scheduling and execution
- Quantization and performance optimizations

