#pragma once

#include "cudex/span.cu.h"
#include <curand_kernel.h>

#include <Eigen/Dense>

namespace alg {

template<size_t N>
__device__ void randomChoice(curandState& state, size_t nElements, size_t (&out)[N]);

template<size_t N>
__device__ void randomChoice(curandState& state, size_t nElements, cudex::DeviceSpan<size_t> out);

template<size_t N>
__host__ __device__
bool selfAdjointInverse(const Eigen::Matrix<float, N, N>& matrix, Eigen::Matrix<float, N, N>& result);

template<typename T>
__host__ __device__ Eigen::Matrix3f xyzAnglesToMatrix(const T& angles);

inline __host__ __device__ Eigen::Vector3f matrixToAnglesXYZ(const Eigen::Matrix3f& m);

}

#include "device_algs.inl.cu.h"
