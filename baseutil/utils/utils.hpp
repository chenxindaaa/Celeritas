#pragma once

#include <cuda_runtime.h>
#include <system_error>
#include <stdarg.h>
#include <numeric>

namespace eUTIL {
    
#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define CUDA_KERNEL_CHECK()          __kernelCheck(__FILE__, __LINE__)
#define LOG(...)                     __log_info(__VA_ARGS__)

template <typename T, typename Tp>
static size_t reduceDims(T begin, T end, Tp init) {
    if (begin >= end) {
        return 0;
    }
    size_t size = std::accumulate(begin, end, init, std::multiplies<>());
    return size;
}

inline static void __cudaCheck(cudaError_t err, const char* file, const int line) 
{
    if (err != cudaSuccess) 
    {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

inline static void __kernelCheck(const char* file, const int line) 
{
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) 
    {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

static void __log_info(const char* format, ...) 
{
    char msg[1000];
    va_list args;
    va_start(args, format);

    vsnprintf(msg, sizeof(msg), format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}

void initMatrix(float* data, int size, int seed);
void printMat(float* data, int size);
void compareMat(float* h_data, float* d_data, int size);

}  // namespace eUTIL