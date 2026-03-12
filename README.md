# Celeritas

轻量 CUDA/C++ 实验工程，当前包含：
- `baseutil`：内存池（CPU/GPU）与 `Tensor` 基础能力
- `celeritas`：示例 CUDA kernel（`add`）
- `test`：基于 gtest 的单元测试（含 pool/tensor/add）

## 1. 依赖

- CMake >= 3.18
- C++17 编译器
- CUDA Toolkit（必需）
- GTest

说明：当前工程强制要求 CUDA，若本机没有 CUDA，CMake 配置会直接失败。

## 2. 目录结构

```text
baseutil/
  memory/Pool.h, Pool.cpp
  tensor/tensor.h, tensor.cpp
celeritas/
  operation/kernels/gpu/add.cuh, add.cu
test/
  testUtil/testPool.cpp
  testOperation/testAdd.cpp
  CMakeLists.txt
```

## 3. 构建与运行测试

在项目根目录执行：

```powershell
cmake -S test -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug --target test_pool
```

运行测试：

```powershell
.\build\debug\test_pool.exe
```

或使用 CTest：

```powershell
ctest --test-dir build/debug --output-on-failure
```

## 4. CUDA 调试（cuda-gdb）

`test/CMakeLists.txt` 已在 `Debug` 下添加 CUDA 调试信息参数：
- `-G`
- `-g`
- `-lineinfo`
- `-Xcompiler=-O0,-g`

建议使用与当前系统匹配的 `cuda-gdb` 环境（如 Linux/WSL + CUDA 工具链）。

## 5. VSCode

工程内已包含 `.vscode/tasks.json` 与 `.vscode/launch.json` 示例配置，可用于：
- 构建 `build/debug/test_pool`
- 启动 gdb 调试 `test_pool`

如调试器路径不同，请修改 `launch.json` 中的 `miDebuggerPath`。

## 6. 当前实现要点

- `MemoryMgr` 为单例，统一管理各类 `MemoryPool`
- `MemoryPool` 使用 size-class freelist 提高复用率
- `Tensor<T>` 目前支持 `int/float/double`
- `Tensor::cpu()` / `Tensor::cuda()` 支持 CPU/GPU 间数据迁移

