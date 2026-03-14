cmake_minimum_required(VERSION 3.18)

project(baseutil LANGUAGES CXX)
enable_language(CUDA)

add_library(baseutil STATIC
  memory/Pool.cpp
  tensor/tensor.cpp
)

target_include_directories(baseutil
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(baseutil PUBLIC cxx_std_17)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(baseutil PUBLIC CUDA::cudart)
