#include "gtest/gtest.h"

#include "octree.cu.h"

#include "timer.h"
#include "cudex/launcher.cu.h"
#include "cudex/device_utils.cu.h"

#include <random>

#include <Eigen/Dense>

using namespace octree;
using namespace cudex;
using namespace core;

namespace {

struct Point
{
    float xval, yval, zval;
};

template<size_t>
__host__ __device__ float get(const Point&);

template<>
__host__ __device__ float get<0>(const Point& p)
{
    return p.xval;
}

template<>
__host__ __device__ float get<1>(const Point& p)
{
    return p.yval;
}

template<>
__host__ __device__ float get<2>(const Point& p)
{
    return p.zval;
}

template<int N>
__global__ void runQuery(Eigen::Matrix<float, N, 3> queries, GPUQuery<Point> q, DeviceSpan<size_t> result)
{
    const auto index = threadLinearIndex();
    if (index >= N) {
        return;
    }

    const Id r = q.findNeighbor(queries.row(index));
    result[index] = r;
}

__global__ void runQueryVec(DeviceSpan<const float3> queries, GPUQuery<float3> q, DeviceSpan<size_t> result)
{
    const auto index = threadLinearIndex();
    if (index >= queries.size()) {
        return;
    }

    const Id r = q.findNeighbor(queries[index]);
    result[index] = r;
}

template<typename Dist, typename Gen>
std::vector<float3> genRandomVector(size_t n, Dist& dist, Gen& gen)
{
    std::vector<float3> ret(n);
    for (auto& v: ret) {
        v.x = dist(gen);
        v.y = dist(gen);
        v.z = dist(gen);
    }

    return ret;
}

size_t findClosest(const float3& query, const std::vector<float3>& points)
{
    auto e = [](const float3& v) {
        return Eigen::Vector3f(v.x, v.y, v.z);
    };

    auto m = std::min_element(points.begin(), points.end(),
        [&](const float3& v1, const float3& v2) {
            return (e(query) - e(v1)).squaredNorm() < (e(query) - e(v2)).squaredNorm();
        }
    );

    return m - points.begin();
}

}

TEST(octree, simple)
{
    constexpr int N = 5;

    const std::vector<Point> points = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
    };

    Eigen::Matrix<float, N, 3> queries;
    queries <<
        0, 0, 0,
        0.4, 0.3, 0.2,
        -4, -4, 2,
        10, 0, 0,
        5, 5, 5;

    const std::vector<size_t> results = { 0, 0, 4, 1, 6 };

    Octree<Point> octree;
    octree.initialize(points);

    const auto gpuq = octree.gpuQuery();
    HostDeviceMemory<size_t> indexMem(N);

    auto launcher = Launcher().sync().size1D(N);
    launcher.run(runQuery<N>, queries, gpuq, indexMem.device());

    indexMem.copyDeviceToHost();

    for (size_t i=0; i < N; ++i) {
        EXPECT_EQ(indexMem[i], results[i]);

        const Point q = { queries(i, 0), queries(i, 1), queries(i, 2) };
        EXPECT_EQ(octree.findNeighbor(q), results[i]);
    }
}

TEST(octree, random)
{
    constexpr size_t N_POINTS = 200000;
    constexpr size_t N_QUERIES = 5331;

    constexpr float POINT_RANGE = 100;
    constexpr float QUERY_RANGE = 120;

    std::uniform_real_distribution<float> pointDist(-POINT_RANGE, POINT_RANGE);
    std::uniform_real_distribution<float> queryDist(-QUERY_RANGE, QUERY_RANGE);
    std::mt19937 gen(0);

    const std::vector<float3> points = genRandomVector(N_POINTS, pointDist, gen);
    const std::vector<float3> queries = genRandomVector(N_QUERIES, queryDist, gen);

    HostDeviceMemory pointsMem(points);
    HostDeviceMemory queriesMem(queries);
    HostDeviceMemory<size_t> resultsMem(N_QUERIES);

    Octree<float3> octree;
    const auto timerInit = Timer();
    octree.initialize(points);
    std::cout << timerInit.printMs("Init time");

    const auto gpuq = octree.gpuQuery();

    auto launcher = Launcher().sync().size1D(N_QUERIES);

    const auto timerCuda = Timer();
    launcher.run(runQueryVec, queriesMem.device(), gpuq, resultsMem.device());
    std::cout << timerCuda.printMs("Cuda time");

    resultsMem.copyDeviceToHost();

    const auto timerCPU = Timer();
    for (size_t i=0; i < N_QUERIES; ++i) {
        const size_t minTree = octree.findNeighbor(queries[i]);
        EXPECT_EQ(resultsMem[i], minTree);
    }
    std::cout << timerCPU.printMs("CPU timer");

    for (size_t i=0; i < 10; ++i) {
        const size_t minLoop = findClosest(queries[i], points);
        EXPECT_EQ(resultsMem[i], minLoop);
    }
}

