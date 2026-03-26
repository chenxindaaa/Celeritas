add_executable(celeritas_story
  story.cpp
)

target_link_libraries(celeritas_story
  PRIVATE
    udm
    celeritas
)

if (EXISTS "/usr/local/cuda/lib64")
  target_link_options(celeritas_story
    PRIVATE
      -L/usr/local/cuda/lib64
  )
endif()

target_compile_options(celeritas_story
  PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g;-O0>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
)

target_include_directories(celeritas_story
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
