cmake_minimum_required(VERSION 3.18)

project(baseutil LANGUAGES CXX)
enable_language(CUDA)

add_library(baseutil_memory STATIC
  memory/Pool.cpp
  tensor/tensor.cpp
)

target_include_directories(baseutil_memory
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(baseutil_memory PUBLIC cxx_std_17)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(baseutil_memory PUBLIC CUDA::cudart)
