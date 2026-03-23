#include <gtest/gtest.h>

#include "celeritas/kernels/KernelFactory.h"
#include "udm/core/Tensor.h"

namespace {

void fillValueStorage(eUTIL::Tensor<float>& storage) {
    const float data[] = {
        1.0f, 2.0f, -99.0f, -99.0f,
        3.0f, 4.0f, -99.0f, -99.0f,
        5.0f, 6.0f
    };

    for (int i = 0; i < 10; ++i) {
        storage[i] = data[i];
    }
}

}  // namespace

TEST(test_scalesum, scalesum_cpu_matches_reference_with_stride) {
    using eUTIL::DeviceType;
    using eUTIL::Tensor;

    Tensor<float> valueStorage(DeviceType::kCpu, 10);
    Tensor<float> scale(DeviceType::kCpu, 3);
    Tensor<float> output(DeviceType::kCpu, 2);

    fillValueStorage(valueStorage);
    scale[0] = 0.5f;
    scale[1] = -1.0f;
    scale[2] = 2.0f;
    output[0] = 0.0f;
    output[1] = 0.0f;

    Tensor<float> value(DeviceType::kCpu, {2}, valueStorage.data(), true);

    auto kernel = eCEL::KernelFactory::getScalesumKernel();
    kernel(value, scale, output, 2, 2, 4, nullptr);

    EXPECT_NEAR(output[0], 7.5f, 1e-5f);
    EXPECT_NEAR(output[1], 9.0f, 1e-5f);
}
