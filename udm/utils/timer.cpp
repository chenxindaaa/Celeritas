#include <chrono>
#include <iostream>
#include <memory>
#include "timer.hpp"

#include "utils.hpp"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

Timer::Timer(){
    m_timeElapsed = 0;
    m_cpuStart = std::chrono::high_resolution_clock::now();
    m_cpuStop = std::chrono::high_resolution_clock::now();
    cudaEventCreate(&m_gpuStart);
    cudaEventCreate(&m_gpuStop);
}

Timer::~Timer(){
    cudaFree(m_gpuStart);
    cudaFree(m_gpuStop);
}

void Timer::startGpu() {
    cudaEventRecord(m_gpuStart, 0);
}

void Timer::stopGpu() {
    cudaEventRecord(m_gpuStop, 0);
}

void Timer::startCpu() {
    m_cpuStart = std::chrono::high_resolution_clock::now();
}

void Timer::stopCpu() {
    m_cpuStop = std::chrono::high_resolution_clock::now();
}

void Timer::durationGpu(std::string msg){
    CUDA_CHECK(cudaEventSynchronize(m_gpuStart));
    CUDA_CHECK(cudaEventSynchronize(m_gpuStop));
    cudaEventElapsedTime(&m_timeElapsed, m_gpuStart, m_gpuStop);

    LOG("%-60s uses %.6lf ms", msg.c_str(), m_timeElapsed);
}
