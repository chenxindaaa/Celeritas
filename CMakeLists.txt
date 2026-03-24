cmake_minimum_required(VERSION 3.18)

project(Celeritas LANGUAGES CXX)
enable_language(CUDA)

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug or Release)" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release)
option(ENABLE_ASAN "Enable AddressSanitizer for C++ targets" OFF)

if (ENABLE_ASAN)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address;-fno-omit-frame-pointer>)
  add_link_options(-fsanitize=address)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(CUDAToolkit REQUIRED)

if (TARGET glog::glog)
  set(CELERITAS_GLOG_TARGET glog::glog)
elseif(TARGET Glog::glog)
  set(CELERITAS_GLOG_TARGET Glog::glog)
else()
  message(FATAL_ERROR "glog target not found after find_package(glog REQUIRED)")
endif()

add_subdirectory(udm)
add_subdirectory(celeritas)
add_subdirectory(demo)
add_subdirectory(test)
