#pragma once

#include "cudex/span.cu.h"
#include "cudex/memory.cu.h"
#include "cudex/cub.cu.h"

#include "ransac.cu.h"
#include "octree.cu.h"

#include <Eigen/Dense>

#include <memory>

#include "icp_types.h"

namespace alg {

namespace icp_impl {
    struct Model;
    struct PointPlane;
    struct Plane;
}

class ICP
{
public:
    constexpr static size_t MIN_POINTS = 30;
    constexpr static size_t MIN_MATCHED_INLIERS = 10;

    using Point = Eigen::Vector3f;

public:
    ICP(const ICPParams& params);
    ~ICP();

    ICPResult run(cudex::HostSpan<const Eigen::Vector3f> src,
            cudex::HostSpan<const Eigen::Vector3f> dst,
            cudex::HostSpan<const Eigen::Vector3f> dstNormals,
            const Eigen::Matrix3f& initialRotation,
            const Eigen::Vector3f& initialTranslation);

    ICPResult run(cudex::HostSpan<const float3> src,
            cudex::HostSpan<const float3> dst,
            cudex::HostSpan<const float3> dstNormals,
            const Eigen::Matrix3f& initialRotation,
            const Eigen::Vector3f& initialTranslation);

private:
    RansacParams ransacParams() const;
    ICPResult resultError(ICPResult::Status) const;

    cudex::DeviceSpan<const icp_impl::PointPlane> transformAndMatch(
            cudex::DeviceSpan<Point> src,
            const Eigen::Affine3f& transformation,
            const octree::GPUQuery<icp_impl::Plane>& gpuQuery);

    octree::GPUQuery<icp_impl::Plane> planeDeviceQuery(
            cudex::HostSpan<const float3> points,
            cudex::HostSpan<const float3> normals);

    cudex::DeviceSpan<Point> copySourcePoints(cudex::HostSpan<const float3> hostSrc);

private:
    using RansacT = Ransac<icp_impl::Model>;

    ICPParams params_;
    std::unique_ptr<RansacT> ransac_;

    cudex::DeviceMemory<float3> pointsSrcMem_;          // Original points from the user
    cudex::DeviceMemory<Point> pointsTransformedMem_;   // Transformed points (in each iteration of ICP)

    cudex::DeviceMemory<icp_impl::PointPlane> pointPlaneAllMem_;
    cudex::DeviceMemory<icp_impl::PointPlane> pointPlaneValidMem_;
    cudex::HostDeviceMemory<icp_impl::PointPlane> inliersMem_;

    cudex::PartitionIf partitionIf_;

    octree::Octree<icp_impl::Plane> octree_;
    std::vector<icp_impl::Plane> planeData_;
};



}
