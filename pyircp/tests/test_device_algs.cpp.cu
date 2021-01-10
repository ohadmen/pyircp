#include "gtest/gtest.h"

#include "device_algs.cu.h"

#include "cudex/launcher.cu.h"
#include "cudex/memory.cu.h"
#include "cudex/device_utils.cu.h"

#include <random>

using namespace cudex;
using namespace alg;

namespace {

template<size_t M>
__global__ void kernelRandomChoice(size_t nElements, size_t nThreads, DeviceSpan<size_t> out)
{
    constexpr int seed = 0;

    const auto index = threadLinearIndex();
    if (index >= nThreads) {
        return;
    }

    curandState state;
    curand_init(seed, index, 0, &state);
    randomChoice<M>(state, nElements, out.subspan(index * M, M));
}

template<size_t N>
using Matrix = Eigen::Matrix<float, N, N>;

template<size_t N>
__global__ void kernelMatrixInverse(DeviceSpan<const Matrix<N>> in, DeviceSpan<Matrix<N>> out)
{
    assert(in.size() == out.size());

    const auto index = threadLinearIndex();
    if (index >= in.size()) {
        return;
    }

    selfAdjointInverse<N>(in[index], out[index]);
}

struct RandomChoiceRunner
{
    template<size_t M>
    HostSpan<size_t> run(size_t nElements, size_t nThreads)
    {
        mem.resize(nThreads * M);

        CHECK_EQ(mem.device().size(), nThreads * M);
        CHECK_EQ(mem.size(), mem.device().size());

        auto launcher = Launcher(nThreads).sync();
        launcher.run(kernelRandomChoice<M>, nElements, nThreads, mem.device());

        mem.copyDeviceToHost();
        return mem.host();
    }
    
    HostDeviceMemory<size_t> mem;
};

template<typename Matrix>
bool closeToIdentity(const Matrix& m)
{
    return ((m -  m.Identity()).array().abs() < 1e-5).all();
}

}

TEST(randomChoice, general)
{
    constexpr size_t N_THREADS = 133;
    constexpr size_t N_ELEMENTS = 1000;
    constexpr size_t N_CHOICES = 10;

    RandomChoiceRunner r;
    r.run<N_CHOICES>(N_ELEMENTS, N_THREADS);

    EXPECT_EQ(N_THREADS * N_CHOICES, r.mem.size());

    for (size_t t = 0; t < N_THREADS; ++t) {
        for (size_t i = 0; i < N_CHOICES - 1; ++i) {
            const size_t& v = r.mem.host()[t*N_CHOICES + i];
            const size_t& vn = r.mem.host()[t*N_CHOICES + i + 1];

            EXPECT_LT(v, vn);
            EXPECT_LT(vn, N_ELEMENTS);
        }
    }
}

TEST(randomChoice, full)
{
    constexpr size_t N_THREADS = 133;
    constexpr size_t N_ELEMENTS = 100;
    constexpr size_t N_CHOICES = N_ELEMENTS;

    RandomChoiceRunner r;
    r.run<N_CHOICES>(N_ELEMENTS, N_THREADS);

    EXPECT_EQ(N_THREADS * N_CHOICES, r.mem.size());

    for (size_t t = 0; t < N_THREADS; ++t) {
        for (size_t i = 0; i < N_CHOICES - 1; ++i) {
            const size_t& v = r.mem.host()[t*N_CHOICES + i];
            EXPECT_EQ(v, i);
        }
    }
}

TEST(randomChoice, single)
{
    constexpr size_t N_THREADS = 133;
    constexpr size_t N_ELEMENTS = 1;
    constexpr size_t N_CHOICES = N_ELEMENTS;

    RandomChoiceRunner r;
    r.run<N_CHOICES>(N_ELEMENTS, N_THREADS);

    EXPECT_EQ(N_THREADS * N_CHOICES, r.mem.size());

    for (size_t t = 0; t < N_THREADS; ++t) {
        EXPECT_EQ(r.mem[t], 0);
    }
}

TEST(matrixInverse, randomMatrixCPU)
{
    constexpr size_t N = 6;
    constexpr size_t M = 20;
    constexpr int SEED = 0;

    std::mt19937 gen(SEED);
    std::uniform_real_distribution<float> dis(-10, 10);

    const Eigen::Matrix<float, M, N> m = Eigen::Matrix<float, M, N>::NullaryExpr([&]() {
       return dis(gen);
    });

    const Eigen::Matrix<float, N, N> mm = m.transpose() * m;

    Eigen::Matrix<float, N, N> res;

    const bool ok = selfAdjointInverse<N>(mm, res);
    EXPECT_TRUE(ok);

    EXPECT_TRUE(closeToIdentity(res * mm));
}

TEST(matrixInverse, randomMatrixGPU)
{
    constexpr size_t N_THREADS = 333;
    constexpr size_t N = 6;
    constexpr size_t M = 15;
    constexpr int SEED = 0;

    std::mt19937 gen(SEED);
    std::uniform_real_distribution<float> dis(-10, 10);

    HostDeviceMemory<Matrix<N>> inMem(N_THREADS);
    HostDeviceMemory<Matrix<N>> outMem(N_THREADS);

    for (auto& m : inMem.host().tail(N_THREADS - 1)) {
        const Eigen::Matrix<float, M, N> tm = Eigen::Matrix<float, M, N>::NullaryExpr([&]() {
            return dis(gen);
        });
        m = tm.transpose() * tm;
    }

    inMem[0] = Eigen::Matrix<float, N, N>::Identity();

    inMem.copyHostToDevice();

    auto launcher = Launcher(N_THREADS).sync();
    launcher.run(kernelMatrixInverse<N>, inMem.device(), outMem.device());

    outMem.copyDeviceToHost();

    for (size_t i = 0; i < N_THREADS; ++i) {
        EXPECT_TRUE(closeToIdentity(outMem[i] * inMem[i]));
    }
}

TEST(rotation, anglesToMatrix)
{
    using namespace Eigen;

    const Vector3f angles(0.146441, -0.278534, 0.645568);

    const Matrix3f matrix = xyzAnglesToMatrix(angles);

    const Vector3f angles2 = matrixToAnglesXYZ(matrix);

    const Matrix3f matrix2 = xyzAnglesToMatrix(angles2);

    constexpr float EPS = 1e-5;

    EXPECT_TRUE(((matrix - matrix2).array() < EPS).all());

    EXPECT_TRUE(((angles - angles2).array() < EPS).all())
        << "Angles: " << angles.transpose() << " - " << angles2.transpose();
}

