#pragma once

#include "udm/core/Tensor.h"
#include "udm/common/cudaConfig.h"

namespace eCEL {
template<typename T>
void mhaKernelCu(int32_t pos, int32_t head_num, 
                  int32_t layer_index, int32_t seq_len, 
                  int32_t kv_dim, int32_t kv_mul, int32_t head_size, 
                  const eUTIL::Tensor<T>& query_tensor, 
                  const eUTIL::Tensor<T>& key_cache_tensor, 
                  const eUTIL::Tensor<T>& value_cache_tensor,
                  eUTIL::Tensor<T>& score_tensor,
                  eUTIL::Tensor<T>& mha_out,
                  const eUTIL::CudaConfig* config);

template<>
void mhaKernelCu(int32_t pos, int32_t head_num, 
                  int32_t layer_index, int32_t seq_len, 
                  int32_t kv_dim, int32_t kv_mul, int32_t head_size, 
                  const eUTIL::Tensor<float>& query_tensor, 
                  const eUTIL::Tensor<float>& key_cache_tensor, 
                  const eUTIL::Tensor<float>& value_cache_tensor,
                  eUTIL::Tensor<float>& score_tensor,
                  eUTIL::Tensor<float>& mha_out,
                  const eUTIL::CudaConfig* config);
}  // namespace eCEL