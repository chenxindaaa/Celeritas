#include "udm/core/Tensor.h"
namespace eCEL {
template<typename T>
void softmaxInplaceCpu(eUTIL::Tensor<T>& input, void* stream = nullptr) 
{
    (void)stream;
    int32_t size = static_cast<int32_t>(input.size());
    // avoid overflow
    float max_value = *std::max_element(input.data(), input.data() + size);

    arma::Col<T> input_mat(const_cast<float*>(input.data()), size, false, true);
    input_mat = arma::exp(input_mat - max_value);

    float sum_value = arma::sum(input_mat);
    input_mat = input_mat / sum_value;
}

// void softmax_inplace_cpu(float* input_ptr, size_t size) {
//     tensor::Tensor input(base::DataType::kDataTypeFp32, size);
//     std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
//         size * sizeof(float), nullptr, (void*)input_ptr, true);
//     input.assign(buffer);
//     return softmax_inplace_cpu(input);
// }
}  // namespace eCEL