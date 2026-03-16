#pragma once

#include <chrono>
#include <ratio>
#include <string>
#include "cuda_runtime.h"

namespace eUTIL {

class Timer {
public:
    using s  = std::ratio<1, 1>;
    using ms = std::ratio<1, 1000>;
    using us = std::ratio<1, 1000000>;
    using ns = std::ratio<1, 1000000000>;

public:
    Timer();
    ~Timer();

public:
    void startCpu();
    void startGpu();
    void stopCpu();
    void stopGpu();

    template <typename span>
    void durationCpu(std::string msg);

    void durationGpu(std::string msg);

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_cpuStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_cpuStop;
    cudaEvent_t m_gpuStart;
    cudaEvent_t m_gpuStop;
    float m_timeElapsed;
};

template <typename span>
void Timer::durationCpu(std::string msg){
    std::string str;

    if(std::is_same<span, s>::value) { str = "s"; }
    else if(std::is_same<span, ms>::value) { str = "ms"; }
    else if(std::is_same<span, us>::value) { str = "us"; }
    else if(std::is_same<span, ns>::value) { str = "ns"; }

    std::chrono::duration<double, span> time = m_cpuStop - m_cpuStart;
    LOG("%-40s uses %.6lf %s", msg.c_str(), time.count(), str.c_str());
}
}  // namespace eUTIL
