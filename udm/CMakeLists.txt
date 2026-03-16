cmake_minimum_required(VERSION 3.18)

project(udm LANGUAGES CXX)
enable_language(CUDA)

add_library(udm STATIC
  memory/Pool.cpp
  memory/MemoryMgr.cpp
  core/Tensor.cpp
)

target_include_directories(udm
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(udm PUBLIC cxx_std_17)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(udm PUBLIC CUDA::cudart)

