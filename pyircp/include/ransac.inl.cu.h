#pragma once

#include "exception.h"

#include "cudex/device_utils.cu.h"
#include "cudex/launcher.cu.h"
#include "cudex/uarray.cu.h"

namespace alg {

// ----- Kernels

namespace ransac_impl
{

using namespace cudex;

template<typename Model>
__global__ void initModels(
        DeviceSpan<curandState> random,
        DeviceSpan<Model> models,
        DeviceSpan<const typename Model::Point> points,
        DeviceSpan<bool> validSpan,
        DeviceSpan<RansacScore> scores,
        size_t attempts,
        const Model* defaultModel)
{
    assert(random.size() == models.size());

    const auto index = threadLinearIndex();
    if (index >= models.size())
    {
        return;
    }

    bool valid;
    size_t cnt = 0;

    Model model = *defaultModel;

    do
    {
        valid = model.init(random[index], points, index);
        ++cnt;
    } while(!valid && cnt < attempts);

    models[index] = model;
    validSpan[index] = valid;

    scores[index].error = 0;
    scores[index].nInliers = 0;
}

template<typename Model, size_t POINT_BLOCK_SIZE>
__global__ void calculateErrors(
        DeviceSpan<const Model> models,
        DeviceSpan<const bool> valid,
        DeviceSpan<const typename Model::Point> points,
        DeviceSpan<RansacScore> scores,
        const float maxError)
{
    __shared__ UArray<typename Model::Point, POINT_BLOCK_SIZE> localPoints;

    const uint3 mIndex = threadMatrixIndex();

    assert(mIndex.z == 0);
    assert(threadIdx.y == 0);
    assert(threadIdx.z == 0);

    const size_t modelIndex = mIndex.x;
    const size_t pointBlockIndex = mIndex.y;

    const size_t pointBeg = pointBlockIndex * POINT_BLOCK_SIZE;
    assert(pointBeg < points.size());

    const size_t pointEnd = min(pointBeg + POINT_BLOCK_SIZE, points.size());
    const size_t nPoints = pointEnd - pointBeg;

    assert(blockDim.x > POINT_BLOCK_SIZE);

    const size_t thread = threadIdx.x;

    assert(nPoints <= POINT_BLOCK_SIZE);
    if (thread < nPoints)
    {
        localPoints[thread] = points[pointBeg + thread];
    }

    if (modelIndex >= models.size())
    {
        return;
    }

    if (!valid[modelIndex])
    {
        return;
    }

    __syncthreads();

    float err = 0;
    size_t nInliers = 0;

    const Model model = models[modelIndex];

    for (size_t i = 0; i < nPoints; ++i)
    {
        const float pointError = model.pointError(localPoints[i]);
        if (pointError < maxError)
        {
            ++nInliers;
            err += pointError;
        }
    }

    atomicAdd(&scores[modelIndex].error, err);
    atomicAdd(&scores[modelIndex].nInliers, nInliers);
}

// Has to be template, because inline on __global__ generates an error

template<typename Seed>
__global__ void initRandom(Seed seed, const DeviceSpan<curandState> states)
{
    const auto index = threadLinearIndex();
    if (index >= states.size())
    {
        return;
    }

    curand_init(seed, index, 0, &states[index]);
}

}

// ----- Class methods

template<typename Model>
Ransac<Model>::Ransac(const RansacParams& params)
    : Ransac(params, Model())
{
}

template<typename Model>
Ransac<Model>::Ransac(const RansacParams& params, const Model& defaultModel)
    : params_(params)
    , defaultModel_(1)
    , modelsMem_(params.nIterations)
    , modelsValidMem_(params.nIterations)
    , scoresMem_(params.nIterations)
    , randomMem_(params.nIterations)
{
    REQUIRE(params.valid(), "Invalid ransac parameters");

    const auto launcher = cudex::Launcher(modelsMem_.size());
    launcher.run(ransac_impl::initRandom<int>,
        params_.seed,
        randomMem_.span()
    );

    defaultModel_.host()[0] = defaultModel;
    defaultModel_.copyHostToDeviceAsync();

    cudex::syncCuda();
}

template<typename Model>
auto Ransac<Model>::run(cudex::DeviceSpan<const Point> points) -> Result
{
    using namespace cudex;

    // Initialize models
    const auto launcher1 = Launcher(modelsMem_.size());
    launcher1.run(ransac_impl::initModels<Model>,
        randomMem_.span(),
        modelsMem_.device(),
        points,
        modelsValidMem_.device(),
        scoresMem_.device(),
        params_.modelAttempts,
        defaultModel_.cdevice().begin()
    );

    // Calculate inliers and errors
    
    static_assert(Launcher::N_BLOCK_THREADS > POINT_BLOCK_SIZE, "calculateErrors algorithm");

    const auto grid = dim3{
        divDim(modelsMem_.size(), Launcher::N_BLOCK_THREADS),
        divDim(points.size(), POINT_BLOCK_SIZE),
        1
    };

    const auto launcher2 = Launcher().sizeGrid(grid);
    launcher2.run(ransac_impl::calculateErrors<Model, POINT_BLOCK_SIZE>,
        modelsMem_.cdevice(),
        modelsValidMem_.cdevice(),
        points,
        scoresMem_.device(),
        params_.inlierMaxError
    );

    modelsMem_.copyDeviceToHost();
    scoresMem_.copyDeviceToHost();

    const RansacScore* bestIt = std::max_element(scoresMem_.host().begin(), scoresMem_.host().end());
    const size_t ind = bestIt - scoresMem_.host().begin();

    return {
        modelsMem_.host()[ind],
        {
            bestIt->nInliers,
            bestIt->error
        }
    };
}

template<typename Model>
auto Ransac<Model>::partitionInliers(
        const Model& model,
        cudex::DeviceSpan<const Point> points,
        cudex::DeviceSpan<Point> result) -> cudex::DeviceSpan<Point>
{
    auto selector = [model, params=params_] __device__ (const Point& point)
    {
        return model.pointError(point) < params.inlierMaxError;
    };

    return partitionIf_.runSync(result, points, selector);
}

}

