cmake_minimum_required(VERSION 3.18)

project(celeritas_gtests LANGUAGES CXX)
enable_language(CUDA)

# Single-config generators (Ninja/Makefiles): default to Debug if not set.
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug or Release)" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release)

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

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../udm
                 ${CMAKE_BINARY_DIR}/udm)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../celeritas
                 ${CMAKE_BINARY_DIR}/celeritas)

add_executable(test_celeritas
  testMain.cpp
  testUtil/testLog.cpp
  testUtil/testPool.cpp
  testUtil/testTensor.cpp
  testUtil/testSave.cpp
  testOperation/testAdd.cpp
  testOperation/testRms.cpp
)

target_link_libraries(test_celeritas
  PRIVATE
    udm
    celeritas
    GTest::gtest
    ${CELERITAS_GLOG_TARGET}
)

target_compile_options(test_celeritas
  PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g;-O0>
)

target_include_directories(test_celeritas
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

enable_testing()
add_test(NAME pool_tests COMMAND test_celeritas)
