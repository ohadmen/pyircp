#include "icp.cu.h"

#include "device_algs.cu.h"
#include "timer.h"

constexpr size_t N_MODEL_POINTS = 6;

using namespace cudex;

using Vector6f = Eigen::Vector<float, 6>;
using Matrix6f = Eigen::Matrix<float, 6, 6>;

using PointMatrix = Eigen::Matrix<float, Eigen::Dynamic, 3>;

namespace {

__host__ __device__ __forceinline__ Eigen::Vector3f INVALID_NORMAL()
{
    return Eigen::Vector3f::Zero();
}

}

// -------------------------------------------------------------------------------------------------
// Impl structs

namespace alg::icp_impl {

// ----- Plane
struct Plane
{
    Eigen::Vector3f point;
    Eigen::Vector3f normal;
};

template<size_t N>
__host__ __device__ float get(const Plane& plane)
{
    return ::octree::get<N>(plane.point);
}

// ----- PointPlane
struct PointPlane
{
    Eigen::Vector3f src;
    Eigen::Vector3f dst;
    Eigen::Vector3f normal;

    size_t srcIndex;

    Vector6f a;
    float b;
};

// ----- Model
struct Model
{
    using Point = PointPlane;

    __device__ bool init(curandState& state, DeviceSpan<const PointPlane> points, size_t index)
    {
        // if (index == 0)
        // {
        //     theta = Vector6f::Zero();
        //     return true;
        // }

        size_t selected[N_MODEL_POINTS];
        randomChoice(state, points.size(), selected);

        Eigen::Matrix<float, N_MODEL_POINTS, 6> A;
        Eigen::Vector<float, N_MODEL_POINTS> b;

        for (size_t i=0; i < N_MODEL_POINTS; ++i) {
            const PointPlane& pp = points[selected[i]];
            A.row(i) = pp.a;
            b[i] = pp.b;
        }

        Matrix6f AadjInv;
        const bool ok = selfAdjointInverse<6>(A.transpose() * A, AadjInv);
        if (!ok) {
            return false;
        }

        Matrix6f Apinv = AadjInv * A.transpose();

        theta = Apinv * b;

        return true;
    }

    __host__ __device__ float pointError(const PointPlane& pp) const
    {
        return abs(pp.a.dot(theta) - pp.b);
    }

    __host__ __device__ Eigen::Affine3f transformation() const
    {
        return Eigen::Translation3f(theta.tail(3)) * xyzAnglesToMatrix(theta.head(3));
    }

    __host__ static Model fromTransformation(const Eigen::Affine3f& t)
    {
        Vector6f theta;
        theta.head(3) = matrixToAnglesXYZ(t.rotation());
        theta.tail(3) = t.translation();

        return {theta};
    }

    __host__ static Model zero()
    {
        return { Vector6f::Zero() };
    }

    Vector6f theta;
};

} // namespace alg::icp_impl

using alg::icp_impl::Model;
using alg::icp_impl::Plane;
using alg::icp_impl::PointPlane;


// -------------------------------------------------------------------------------------------------
// Internal methods and kernels

namespace alg {

namespace {

using Point = ICP::Point;

__host__ __device__ auto toEigen(const float3& v)
{
    static_assert(sizeof(Point) == sizeof(float3));
    return Eigen::Map<const Point>(reinterpret_cast<const float*>(&v));
}

__host__ __device__ bool isValidValue(const Eigen::Vector3f& point)
{
    return point.array().isFinite().all();
}

__host__ __device__ bool isValidValue(const PointPlane& pp)
{
    assert(
        pp.normal == INVALID_NORMAL() ||
        isValidValue(pp.dst) && isValidValue(pp.src)
    );
    return pp.normal != INVALID_NORMAL();
}

__global__ void transformPoints(DeviceSpan<Point> points, const Eigen::Matrix3f R, const Eigen::Vector3f t)
{
    const auto index = threadLinearIndex();
    if (index >= points.size())
    {
        return;
    }

    const Point& point = points[index];

    if (isValidValue(point)) {
        points[index] = R * point + t;
    }
}


__global__ void makePointPlane(
        DeviceSpan<const Point> points,
        octree::GPUQuery<Plane> gpuQuery,
        const float maxDistance,
        DeviceSpan<PointPlane> out)
{
    assert(points.size() <= out.size());
    assert(gpuQuery.nPoints() > 0);

    const auto index = threadLinearIndex();
    if (index >= points.size())
    {
        return;
    }

    PointPlane pp;

    pp.src = points[index];
    pp.srcIndex = index;

    pp.normal = INVALID_NORMAL();

    if (isValidValue(pp.src)) {
        const size_t dstId = gpuQuery.findNeighbor(pp.src);

        assert(dstId != octree::INVALID_ID);
        const Plane& plane = gpuQuery.point(dstId);

        pp.dst = plane.point;

        const float dist2 = (pp.src - pp.dst).squaredNorm();

        if (dist2 < maxDistance * maxDistance) {
            pp.normal = plane.normal;

            pp.a.head(3) = pp.src.cross(pp.normal);
            pp.a.tail(3) = pp.normal;

            pp.b = - pp.normal.dot(pp.src - pp.dst);
        }

        assert(isValidValue(pp.dst));
    }

    out[index] = pp;
}


Eigen::Affine3f findRigidBodyTransform(const PointMatrix& src, const PointMatrix& dst)
{
    using namespace Eigen;

    CHECK_EQ(src.rows(), dst.rows());

    const Vector3f srcMean = src.colwise().mean();
    const Vector3f dstMean = dst.colwise().mean();

    const PointMatrix srcN = src.rowwise() - srcMean.transpose();
    const PointMatrix dstN = dst.rowwise() - dstMean.transpose();

    const Matrix3f S = srcN.transpose() * dstN;
    const JacobiSVD<Matrix3f> svd(S, ComputeFullU | ComputeFullV);

    const auto& U = svd.matrixU();
    const auto& V = svd.matrixV();

    const float sgn = (U * V.transpose()).determinant();
    const Vector3f diag(1, 1, sgn);

    const Matrix3f R = V * diag.asDiagonal() * U.transpose();
    const Vector3f t = dstMean - R * srcMean;

    return Translation3f(t) * R;
}


struct ValidPred
{
    template<typename T>
    __device__ bool operator()(const T& v)
    {
        return isValidValue(v);
    }
};

}

// -------------------------------------------------------------------------------------------------
// ICP class

ICP::ICP(const ICPParams& params)
    : params_(params)
    , ransac_(std::make_unique<RansacT>(ransacParams()))
{}

RansacParams ICP::ransacParams() const
{
    RansacParams rp;

    rp.nIterations = params_.ransacIterations;
    rp.inlierMaxError = params_.ransacMaxInlierDistance;
    rp.modelAttempts = 3;

    rp.seed = params_.seed;

    return rp;
}

ICPResult ICP::run(HostSpan<const Point> hostSrc,
        HostSpan<const Point> hostDst,
        HostSpan<const Point> hostNormals,
        const Eigen::Matrix3f& initialRotation,
        const Eigen::Vector3f& initialTranslation)
{
    return run(hostSrc.cast<const float3>(),
        hostDst.cast<const float3>(),
        hostNormals.cast<const float3>(),
        initialRotation,
        initialTranslation);
}

ICPResult ICP::run(const HostSpan<const float3> hostSrc,
        const HostSpan<const float3> hostDst,
        const HostSpan<const float3> hostNormals,
        const Eigen::Matrix3f& initialRotation,
        const Eigen::Vector3f& initialTranslation)
{
    REQUIRE(hostDst.size() == hostNormals.size(), "dst and dstNormals must have same size");

    if (hostSrc.size() < MIN_POINTS) {
        return resultError(ICPResult::TOO_FEW_INPUT_POINTS);
    }

    if (hostDst.size() < MIN_POINTS) {
        return resultError(ICPResult::TOO_FEW_INPUT_POINTS);
    }

    core::Timer timer;
    const auto planeQuery = planeDeviceQuery(hostDst, hostNormals);
    VLOG(2) << timer.printMs("ICP octree");

    const DeviceSpan<Point> srcTransformed = copySourcePoints(hostSrc);
    VLOG(2) << timer.printMs("Init: source copy");

    const Eigen::Affine3f initialTranformation = Eigen::Translation3f(initialTranslation) * initialRotation;
    DeviceSpan<const PointPlane> current = transformAndMatch(
        srcTransformed,
        initialTranformation,
        planeQuery
    );
    VLOG(2) << timer.printMs("Init: transformAndMatch");
    VLOG(2) << "Initial inliers: " << current.size();

    if (current.size() < MIN_MATCHED_INLIERS) {
        return resultError(ICPResult::TOO_FEW_MATCHED_POINTS);
    }

    Model bestModel = Model::fromTransformation(initialTranformation);

    RansacScore bestScore;
    size_t nonIncreaseCnt = 0;
    DeviceSpan<const PointPlane> inliersDev;

    VLOG(2) << timer.printMs("ICP init done");

    for (size_t iter = 0; iter < params_.maxIterations && current.size() >= MIN_MATCHED_INLIERS; ++iter)
    {
        VLOG(2) << "ICP iteration: " << iter << ", current theta: " << bestModel.theta.transpose();
        VLOG(2) << timer.printMs("Loop");

        const auto currentResult = ransac_->run(current);

        VLOG(2) << "    Ransac result: " << currentResult.model.theta.transpose();
        VLOG(2) << "    inliers: " << currentResult.score.nInliers << ", error: " << currentResult.score.error;

        if (bestScore < currentResult.score) {
            bestScore = currentResult.score;

            nonIncreaseCnt = 0;

            inliersDev = ransac_->partitionInliers(currentResult.model, current, inliersMem_.device());
            CHECK_EQ(inliersDev.size(), bestScore.nInliers);

            const Eigen::Affine3f t = currentResult.model.transformation();

            current = transformAndMatch(srcTransformed, t, planeQuery);

            if (VLOG_IS_ON(2)) {
                bestModel = Model::fromTransformation(t * bestModel.transformation());
            }

            VLOG(2) << "    Updating the best result";
        } else {
            ++nonIncreaseCnt;
            if (nonIncreaseCnt == params_.maxNonIncreaseIterations) {
                break;
            }
        }
    }
    VLOG(2) << timer.printMs("ICP loop finished");
    VLOG(2) << "Best inliers count: " << bestScore.nInliers;

    ICPResult ret;

    if (bestScore.nInliers == 0) {
        return resultError(ICPResult::RANSAC_FAILED);
    }

    auto inliers = inliersMem_.copyDeviceToHost(inliersDev.size());

    CHECK_GT(inliers.size(), 0);
    CHECK_EQ(inliersDev.size(), inliers.size());
    CHECK_EQ(inliers.size(), bestScore.nInliers);

    ret.inliers.resize(inliers.size());

    PointMatrix srcMatrix(inliers.size(), 3);
    PointMatrix dstMatrix(inliers.size(), 3);

    size_t ind = 0;
    for (const auto& pp: inliers) {
        assert(isValidValue(pp));

        const size_t hostSrcIdx = pp.srcIndex;

        srcMatrix.row(ind) = toEigen(hostSrc[hostSrcIdx]);
        dstMatrix.row(ind) = pp.dst;

        ret.inliers[ind] = hostSrcIdx;

        ++ind;
    }

    const Eigen::Affine3f t = findRigidBodyTransform(srcMatrix, dstMatrix);

    ret.rotation = t.rotation();
    ret.translation = t.translation();
    ret.nInliers = inliers.size();

    VLOG(2) << timer.printMs("ICP finished");

    ret.status = ICPResult::OK;
    return ret;
}

cudex::DeviceSpan<const PointPlane> ICP::transformAndMatch(
        DeviceSpan<Point> src,
        const Eigen::Affine3f& t,
        const octree::GPUQuery<Plane>& gpuQuery)
{
    pointPlaneAllMem_.resize(src.size());
    pointPlaneValidMem_.resize(src.size());
    inliersMem_.resize(src.size());

    const auto launcher = cudex::Launcher(src.size()).async();

    launcher.run(transformPoints, src, t.rotation(), t.translation());

    launcher.run(makePointPlane,
        src,
        gpuQuery,
        params_.maxMatchingDistance,
        pointPlaneAllMem_.span()
    );

    const DeviceSpan<const PointPlane> ret =  partitionIf_.runSync(
        pointPlaneValidMem_.span(),
        pointPlaneAllMem_.cspan(),
        ValidPred()
    );

    return ret;
}

DeviceSpan<Point> ICP::copySourcePoints(HostSpan<const float3> hostSrc)
{
    DeviceSpan<const float3> src = pointsSrcMem_.resizeCopy(hostSrc);

    // Treats float3 struct as Eigen::Vector3f
    // Device memory is always properly aligned
    return pointsTransformedMem_.resizeCopy(src.cast<const Point>());
}

octree::GPUQuery<Plane> ICP::planeDeviceQuery(
        HostSpan<const float3> points,
        HostSpan<const float3> normals)
{
    CHECK_EQ(points.size(), normals.size());

    planeData_.clear();

    for (size_t i = 0; i < points.size(); ++i) {
        const Point& point = toEigen(points[i]);
        const Point& normal = toEigen(normals[i]);

        if (!isValidValue(point) || !isValidValue(normal) || normal == INVALID_NORMAL()) {
            continue;
        }

        planeData_.emplace_back(Plane({point, normal.normalized()}));
    }

    octree_.initialize(planeData_);

    return octree_.gpuQuery();
}

ICPResult ICP::resultError(ICPResult::Status code) const
{
    CHECK(code != ICPResult::OK);

    ICPResult result;
    result.status = code;
    result.nInliers = 0;

    return result;
}

ICP::~ICP() = default;

}
