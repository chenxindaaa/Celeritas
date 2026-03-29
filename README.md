# Celeritas

Celeritas is a C++ large language model (LLM) inference framework.

It is designed around a clear separation of:

- `udm`: universal data model and foundational runtime utilities (memory pool, tensor, CUDA helpers)
- `celeritas`: model-side operators and kernels
- `test`: unit and integration tests

## Goals

- Build an efficient and extensible C++ inference runtime
- Support heterogeneous backends (CPU/CUDA today, more backends later)
- Keep kernel dispatch and tensor/memory abstractions modular

## Repository Layout

```text
udm/
  common/           # shared device/dtype definitions
  core/             # tensor abstraction
  memory/           # memory pools and memory manager
  utils/            # utility helpers
  CMakeLists.txt

celeritas/
  loader/           # model loader and checkpoint parsing
  kernels/          # kernel implementations + kernel factory
  operation/        # higher-level ops (ongoing)
  kvCache/          # KV cache manager and per-layer cache views
  ropeCache/        # precomputed RoPE sin/cos cache
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
- `Loader` now creates ready-to-run models from checkpoints before they are passed to the engine
- `LLMEngine` now owns a lightweight KV cache manager for per-layer cache allocation
- `MlpLayer` now produces only the MLP result, while residual adds stay in outer blocks such as `DecoderLayer`
- `LLMEngine` now also owns a lightweight RoPE cache so attention can reuse precomputed sin/cos tables
- RoPE cache values are now generated through registered CPU/CUDA kernels instead of hardcoded loops in the cache class
- A new `SelfAttentionLayer` now combines q/k/v/o projections, RoPE, KV cache writes, and MHA into one block
- `DecoderLayer` now uses `SelfAttentionLayer`, so the decoder block matches the standard Llama2 structure more closely
- `Llama2` now builds its decoder layers directly from the checkpoint weight layout during loading
- `Loader` and `Llama2` now support both the old legacy checkpoint layout and the newer `version1_export` layout

## Search Record

- `https://skills.sh/`: checked for an existing KV cache workflow entry, but there was no direct drop-in design for the current C++ engine layout.
- GitHub search: common implementations keep one preallocated K/V buffer per layer and expose lightweight layer views to the runtime.
  References:
  `https://github.com/ggerganov/llama.cpp`
  `https://github.com/vllm-project/vllm`

## Completed And TODO

- Completed: basic `KVCacheManager` structure, per-layer `KVCacheView`, and engine-side allocation hookup.
- Completed: basic `RopeCache` structure and engine-side precomputation hookup.
- Completed: split RoPE cache precomputation into standalone CPU/CUDA kernels and registered them in the kernel factory.
- Completed: added a reusable `SelfAttentionLayer` built on existing matmul, rope, and mha kernels.
- Completed: switched `DecoderLayer` from raw `MhaLayer` to `SelfAttentionLayer`.
- Completed: implemented `Llama2::createDecoders()` to materialize decoder layers from checkpoint weights.
- Completed: aligned loader-side header parsing and Llama2 weight offsets with `version1_export`, while keeping legacy checkpoints usable.
- Completed: moved checkpoint loading out of `Model` into `Loader`, and switched engine/demo/tests to use loaded models.
- Completed: unified decoder-side MLP usage by keeping only `MlpLayer` and moving residual add back to `DecoderLayer`.
- TODO: connect real attention layer writes into KV cache append flow.
- TODO: support cache reuse across multi-step decode instead of resetting per request.

## Roadmap

- More LLM kernels (matmul/rmsnorm/rope/mha/softmax)
- Unified backend registration mechanism (CPU/CUDA/OpenCL...)
- Graph-level scheduling and execution
- Quantization and performance optimizations
