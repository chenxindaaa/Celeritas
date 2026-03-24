#pragma once

namespace eCEL {
size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream);
}