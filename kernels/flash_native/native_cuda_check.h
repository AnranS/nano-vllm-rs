#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

inline void nvr_flash_check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
#define C10_CUDA_CHECK(expression) nvr_flash_check_cuda(expression)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() nvr_flash_check_cuda(cudaGetLastError())
