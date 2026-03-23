#include "../KernelFactory.h"
#include "udm/common/cudaConfig.h"
#include "udm/core/Tensor.h"
namespace eCEL {

// [bs, head_num, seq_len, head_size]
// query_tensor [1, head_num, 1, head_size]
// key_cache_tensor [1, head_num, seq_len, head_size]
// kv_mul default = 1
template <typename T>
void mhaKernelCpu(int32_t pos, int32_t head_num, 
                  int32_t layer_index, int32_t seq_len, 
                  int32_t kv_dim, int32_t kv_mul, int32_t head_size, 
                  const eUTIL::Tensor<T>& query_tensor, 
                  const eUTIL::Tensor<T>& key_cache_tensor, 
                  const eUTIL::Tensor<T>& value_cache_tensor,
                  eUTIL::Tensor<T>& score_tensor,
                  eUTIL::Tensor<T>& mha_out,
                  const eUTIL::CudaConfig* config) {
    int32_t layer_offset = layer_index * seq_len * kv_dim;
    float scale = 1.f / std::sqrt(static_cast<float>(head_size));

    for (int32_t h = 0; h < head_num; ++h) {
        T* score_head_addr = score_tensor.data() + h * seq_len;
        const T* query_head_addr = query_tensor.data() + h * head_size;

        // query_mat [1, 1, 1, head_size]
        const eUTIL::Tensor<T> query_mat(eUTIL::DeviceType::kCpu, {static_cast<std::size_t>(head_size)}, const_cast<T*>(query_head_addr), true/*isExternal*/);

        for (int32_t t = 0; t <= pos; t++) {
            int32_t cache_offset = t * kv_dim + (h / kv_mul) * head_size;
            const float* key_head_addr = key_cache_tensor.data() + layer_offset + cache_offset;
            // key_mat [1, 1, 1, head_size]
            const eUTIL::Tensor<T> key_mat(eUTIL::DeviceType::kCpu, {1, static_cast<std::size_t>(head_size)}, const_cast<T*>(key_head_addr), true/*isExternal*/);

            // score_mat [1]
            eUTIL::Tensor<T> score_mat(eUTIL::DeviceType::kCpu, {1}, score_head_addr + t, true/*isExternal*/);
            KernelFactory::getMatmulKernel()(query_mat, key_mat, score_mat, scale, config);
        }

        // score_head_tensor [pos+1]
        eUTIL::Tensor<T> score_head_tensor(eUTIL::DeviceType::kCpu, {static_cast<std::size_t>(pos) + 1}, score_head_addr, true);

        KernelFactory::getSoftmaxKernel()(score_head_tensor, config ? config->stream : nullptr);

        T* output_head_ptr = mha_out.data() + h * head_size;
        // output_tensor [1, 1, 1, head_size]
        eUTIL::Tensor<T> output_tensor(eUTIL::DeviceType::kCpu, {static_cast<std::size_t>(head_size)}, output_head_ptr, true/*isExternal*/);
        
        int32_t cache_offset = (h / kv_mul) * head_size;
        const T* value_head_addr = value_cache_tensor.data() + layer_offset + cache_offset;
        // value_tensor [1, 1, 1, head_size]
        const eUTIL::Tensor<T> value_tensor(eUTIL::DeviceType::kCpu, {static_cast<std::size_t>(head_size)}, const_cast<T*>(value_head_addr), true/*isExternal*/);
        KernelFactory::getScalesumKernel()(value_tensor, score_head_tensor, output_tensor, pos,
                                           head_size, kv_dim, config ? config->stream : nullptr);
    }
}
}  // namespace eCEL
