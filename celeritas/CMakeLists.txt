cmake_minimum_required(VERSION 3.18)

find_path(SENTENCEPIECE_INCLUDE_DIR
  NAMES sentencepiece_processor.h
  HINTS /usr/local/include /usr/include
)
find_library(SENTENCEPIECE_LIBRARY
  NAMES sentencepiece
  HINTS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
)

add_library(celeritas STATIC
  loader/Loader.cpp
  models/Model.cpp
  models/Llama2.cpp
  engine/LLMEngine.cpp
  kvCache/kvCacheMgr.cpp
  ropeCache/RopeCache.cpp
  sampler/sampler.cpp
  sampler/argmax_sampler.cpp
  tokenizer/Tokenizer.cpp
  tokenizer/BpeTokenizer.cpp
  tokenizer/SentencePieceTokenizer.cpp
  tokenizer/HfTokenizerAdapter.cpp
  kernels/KernelRegistry.cpp
  kernels/KernelFactory.cpp
  operation/Layer.cpp
  operation/AddLayer.cpp
  operation/DecoderLayer.cpp
  operation/EmbLayer.cpp
  operation/MatmulLayer.cpp
  operation/MhaLayer.cpp
  operation/MlpLayer.cpp
  operation/RmsnormLayer.cpp
  operation/SelfAttentionLayer.cpp
  operation/SwigluLayer.cpp
  kernels/gpu/add.cu
  kernels/gpu/argmax_kernel.cu
  kernels/gpu/emb_kernel.cu
  kernels/gpu/matmul_kernel.cu
  kernels/gpu/rmsnorm_kernel.cu
  kernels/gpu/swiglu_kernel.cu
  kernels/gpu/mha_kernel.cu
  kernels/gpu/rope_kernel.cu
  kernels/gpu/rope_cache_kernel.cu
)

target_include_directories(celeritas
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(celeritas PUBLIC cxx_std_17)

find_package(CUDAToolkit REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(celeritas
  PUBLIC
    udm
    CUDA::cudart
    ${SENTENCEPIECE_LIBRARY}
    ${ARMADILLO_LIBRARIES}
    BLAS::BLAS
    LAPACK::LAPACK
)

target_include_directories(celeritas
  SYSTEM PUBLIC
    ${SENTENCEPIECE_INCLUDE_DIR}
    ${ARMADILLO_INCLUDE_DIRS}
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
