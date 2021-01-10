#pragma once

#include "cudex/utils.h"

namespace alg {

template<size_t N>
__device__ void randomChoice(curandState& state, size_t nElements, size_t (&out)[N])
{
    randomChoice<N>(state, nElements, cudex::DeviceSpan<size_t>(out, N));
}

template<size_t N>
__device__ void randomChoice(curandState& state, size_t nElements, cudex::DeviceSpan<size_t> out)
{
    static_assert(N > 0);
    assert(N <= out.size());
    assert(N <= nElements);

    for (size_t i=0; i < N; ++i) {
        size_t ind = curand(&state) % (nElements - i);

        size_t pos = 0;
        while (pos < i && ind >= out[pos]) {
            ++pos;
            ++ind;
        }

        out[i] = ind;

        for (size_t j = i; j > pos; --j) {
            cudex::swap(out[j], out[j-1]);
        }
    }
}

template<size_t N>
__host__ __device__
bool selfAdjointInverse(const Eigen::Matrix<float, N, N>& matrix, Eigen::Matrix<float, N, N>& result)
{
    constexpr float MIN_EIGENVALUE = 1e-4;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, N, N>> solver;
    solver.compute(matrix);

    if (solver.info() != Eigen::Success) {
        return false;
    }

    const float minValue = solver.eigenvalues().array().abs().minCoeff();
    if (minValue < MIN_EIGENVALUE) {
        return false;
    }

    const Eigen::Vector<float, N> diagInverse = solver.eigenvalues().array().inverse();
    result = solver.eigenvectors() * diagInverse.asDiagonal() * solver.eigenvectors().transpose();

    return true;
}

template<typename T>
__host__ __device__ Eigen::Matrix3f xyzAnglesToMatrix(const T& angles)
{
    return (
        Eigen::AngleAxisf(angles[2], Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(angles[1], Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(angles[0], Eigen::Vector3f::UnitX())
    ).toRotationMatrix();
}

__host__ __device__ Eigen::Vector3f matrixToAnglesXYZ(const Eigen::Matrix3f& m)
{
    const Eigen::Vector3f v = m.eulerAngles(2, 1, 0);
    return Eigen::Vector3f(v[2], v[1], v[0]);
}

}
