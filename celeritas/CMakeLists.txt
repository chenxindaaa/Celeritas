cmake_minimum_required(VERSION 3.18)

add_library(celeritas STATIC
  kernels/KernelRegistry.cpp
  kernels/KernelFactory.cpp
  kernels/gpu/add.cu
  kernels/gpu/rmsnorm_kernel.cu
)

target_include_directories(celeritas
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(celeritas PUBLIC cxx_std_17)

find_package(CUDAToolkit REQUIRED)
find_package(Armadillo REQUIRED)
target_link_libraries(celeritas
  PUBLIC
    CUDA::cudart
)

set_target_properties(celeritas PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
)

target_compile_options(celeritas
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-g;-O0>
    $<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g;-O0>
)
