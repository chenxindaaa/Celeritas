add_executable(test_celeritas
  testMain.cpp
  testUtil/testLog.cpp
  testUtil/testPool.cpp
  testUtil/testTensor.cpp
  testUtil/testSave.cpp
  testOperation/testAdd.cpp
  testOperation/testEmb.cpp
  testOperation/testLayer.cpp
  testOperation/testMatmul.cpp
  testOperation/testMha.cpp
  testOperation/testRms.cpp
  testOperation/testScalesum.cpp
  testOperation/testSwiglu.cpp
  testOperation/testSoftmap.cpp
  testModel/testTokenizer.cpp
  testModel/testLlama2.cpp
)

target_link_libraries(test_celeritas
  PRIVATE
    udm
    celeritas
    GTest::gtest
    ${CELERITAS_GLOG_TARGET}
)

if (EXISTS "/usr/local/cuda/lib64")
  target_link_directories(test_celeritas
    PRIVATE
      /usr/local/cuda/lib64
  )
endif()

if (EXISTS "/usr/local/cuda/lib64/libcudadevrt.a" AND EXISTS "/usr/local/cuda/lib64/libcudart_static.a")
  target_link_libraries(test_celeritas
    PRIVATE
      /usr/local/cuda/lib64/libcudadevrt.a
      /usr/local/cuda/lib64/libcudart_static.a
  )
endif()

if (EXISTS "/usr/local/cuda/lib64")
  target_link_options(test_celeritas
    PRIVATE
      -L/usr/local/cuda/lib64
  )
endif()

target_compile_options(test_celeritas
  PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g;-O0>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
)

target_include_directories(test_celeritas
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

enable_testing()
add_test(NAME pool_tests COMMAND test_celeritas)
