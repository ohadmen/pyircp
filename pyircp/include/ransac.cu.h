#pragma once

#include "cudex/span.cu.h"
#include "cudex/memory.cu.h"
#include "cudex/cub.cu.h"

#include <curand_kernel.h>

namespace alg {

struct RansacParams
{
    constexpr static size_t MIN_ITERATIONS = 50;

    size_t nIterations = 0;
    size_t modelAttempts = 0;
    float  inlierMaxError = 0;

    int seed = 0;

    bool valid() const
    {
        return nIterations > MIN_ITERATIONS && modelAttempts > 0 && inlierMaxError > 0;
    }
};

struct RansacScore
{
    uint32_t nInliers = 0;
    float error = 0;

    bool operator<(const RansacScore& other) const
    {
        return nInliers < other.nInliers || nInliers == other.nInliers && error > other.error;
    }
};

template<typename Model>
class Ransac
{
public:
    using Point = typename Model::Point;

    struct Result
    {
        Model model;
        RansacScore score;
    };

public:
    Ransac(const RansacParams& params);
    Ransac(const RansacParams& params, const Model& defaultModel);

    Result run(cudex::DeviceSpan<const Point> points);

    cudex::DeviceSpan<Point> partitionInliers(
        const Model& model,
        cudex::DeviceSpan<const Point> points,
        cudex::DeviceSpan<Point> result);

private:
    inline static constexpr size_t POINT_BLOCK_SIZE = 128;

private:
    cudex::HostDeviceMemory<Model> defaultModel_;
    cudex::HostDeviceMemory<Model> modelsMem_;
    cudex::HostDeviceMemory<bool> modelsValidMem_;
    cudex::HostDeviceMemory<RansacScore> scoresMem_;
    cudex::DeviceMemory<curandState> randomMem_;

    const RansacParams params_;

    cudex::PartitionIf partitionIf_;
};

}

#include "ransac.inl.cu.h"
