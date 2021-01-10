#include "gtest/gtest.h"

#include "ransac.cu.h"

#include "timer.h"

#include <random>

#include <Eigen/Dense>

using namespace alg;
using namespace cudex;

namespace {

struct FitLine
{
    using Point = Eigen::Vector3f;

    __device__ bool init(curandState& state, DeviceSpan<const Point> points, size_t)
    {
        p1_ = points[curand(&state) % points.size()];
        p2_ = points[curand(&state) % points.size()];

        constexpr float MIN_DISTANCE = 0.2;
        return (p1_.head(2) - p2_.head(2)).norm() > MIN_DISTANCE;
    }

    __host__ __device__ float pointError(const Point& point) const
    {
        const Eigen::Vector2f p1 = p1_.head(2);    
        const Eigen::Vector2f p2 = p2_.head(2);

        const Eigen::Vector2f line = (p2 - p1).normalized();
        const Eigen::Vector2f proj = p1 + line.dot(point.head(2) - p1) * line;

        return (point.head(2) - proj).norm();
    }

    Eigen::Vector2f calculateAB() const
    {
        const Eigen::Vector2f p1 = p1_.head(2);
        const Eigen::Vector2f p2 = p2_.head(2);

        const float a = (p1.y() - p2.y()) / (p1.x() - p2.x());
        const float b = p1.y() - a * p1.x();

        return {a, b};
    }

    Point p1_;
    Point p2_;
};

struct DummyModel
{
    using Point = size_t;
    constexpr static size_t BEST_MODEL = 100;
    constexpr static size_t BEST_POINT = 500;

    __device__ bool init(curandState&, DeviceSpan<const Point> points, size_t modelIndex)
    {
        index = modelIndex;
        nPoints = points.size();

        ++tries;

        return index > 10 && tries == 2;
    }

    __device__ float pointError(const Point& point) const
    {
        const size_t indexError = divCeil(absDiff(point, BEST_POINT), 1000);
        const float baseError = indexError + 0.1;

        return index == BEST_MODEL ? baseError : 2 * baseError;
    }

    size_t index;
    size_t tries;
    size_t test;
    size_t nPoints;
};

}

TEST(ransac, fit_line_single)
{
    constexpr float A = -0.34;
    constexpr float B = 2.23;

    constexpr size_t N = 50000;
    constexpr float X_RANGE = 1.12;
    constexpr float NOISE_RANGE = 0.06;

    std::uniform_real_distribution<float> noise(-NOISE_RANGE, NOISE_RANGE);
    std::mt19937 gen(0);

    HostDeviceMemory<Eigen::Vector3f> points(N+1);

    for (size_t i=0; i <= N; ++i)
    {
        const float x = -X_RANGE + i*2*X_RANGE / N;
        const float y = A*x + B;

        points.host()[i] = Eigen::Vector3f{x + noise(gen), y + noise(gen), (float) i};
    }

    points.copyHostToDevice();

    RansacParams params;
    params.nIterations = 3000;
    params.modelAttempts = 5;
    params.inlierMaxError = NOISE_RANGE;

    Ransac<FitLine> ransac(params);

    const auto timerRun = core::Timer();
    const auto res = ransac.run(points.device());
    VLOG(1) << timerRun.printMs("Ransac time");

    const auto ab = res.model.calculateAB();

    EXPECT_NEAR(ab[0], A, 0.01);
    EXPECT_NEAR(ab[1], B, 0.01);
    EXPECT_GT(res.score.nInliers, 4000);

    VLOG(1) << "Inliers: " << res.score.nInliers;
    VLOG(1) << "Result: " << ab.transpose();
    VLOG(1) << "Error: " << (ab - Eigen::Vector2f(A, B)).transpose();

    HostDeviceMemory<Eigen::Vector3f> pointsOut(N+1);

    DeviceSpan<const Eigen::Vector3f> inliersDev = ransac.partitionInliers(res.model, points.device(), pointsOut.device());
    EXPECT_EQ(res.score.nInliers, inliersDev.size());

    pointsOut.copyDeviceToHost();
    auto inliers = pointsOut.host().head(inliersDev.size());
    EXPECT_EQ(inliers.size(), inliersDev.size());

    Eigen::Vector3f prev(-100, -100, -100);
    size_t cnt = 0;
    for (const Eigen::Vector3f& point : inliers)
    {
        EXPECT_LT(res.model.pointError(point), params.inlierMaxError);
        EXPECT_LT(prev.z(), point.z());
        prev = point;

        EXPECT_NEAR(point.x(), -X_RANGE + point.z()*2*X_RANGE / N, NOISE_RANGE);
        ++cnt;
    }
    EXPECT_EQ(cnt, res.score.nInliers);

    auto outliers = pointsOut.host().tail(N + 1 - inliers.size());

    for (const Eigen::Vector3f& point : outliers)
    {
        EXPECT_GE(res.model.pointError(point), params.inlierMaxError);
        EXPECT_NEAR(point.x(), -X_RANGE + point.z()*2*X_RANGE / N, NOISE_RANGE) << point.transpose();
    }
}

TEST(ransac, dummy_model)
{
    constexpr size_t N = 10000;

    HostDeviceMemory<size_t> points(N);
    std::iota(points.host().begin(), points.host().end(), 0);

    points.copyHostToDevice();

    RansacParams params;
    params.nIterations = 5000;
    params.modelAttempts = 5;
    params.inlierMaxError = 6;

    constexpr size_t TEST_VALUE = 55811;

    const DummyModel defaultModel = { 101000, 0, TEST_VALUE };

    Ransac<DummyModel> ransac(params, defaultModel);
    const auto res = ransac.run(points.device());

    EXPECT_EQ(res.model.test, TEST_VALUE);
    EXPECT_EQ(res.model.index, 100);

    EXPECT_EQ(res.score.nInliers, 5501);
    constexpr float expectedError = 0.1 + (1500 * 1.1) + (1000 * 2.1) +
        (1000 * 3.1) + (1000 * 4.1) + (1000 * 5.1);

    EXPECT_NEAR(res.score.error, expectedError, 1e-2);
}

