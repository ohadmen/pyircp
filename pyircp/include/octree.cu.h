#pragma once

#include "cudex/memory.cu.h"
#include "cudex/span.cu.h"

#include <Eigen/Dense>

#include <memory>
#include <type_traits>

namespace octree {

namespace impl {
class Octant;

template<typename Point>
class POctree;
}

using Id = uint32_t;
constexpr inline Id INVALID_ID = static_cast<Id>(-1);

template<typename Point>
class GPUQuery
{
public:
    using Octant = impl::Octant;

    __device__ GPUQuery(cudex::DeviceSpan<const Point>, cudex::DeviceSpan<const Octant>, cudex::DeviceSpan<const Id>);

    __device__ Id findNeighbor(const Point& query, float minDistance = -1) const;

    template<typename Derived>
    __device__ Id findNeighbor(const Eigen::MatrixBase<Derived>& point, float minDistance = -1) const;

    __device__ const Point& point(Id index) const;

    __device__ size_t nPoints() const;

private:
    __device__ Id findNeighborInternal(const Eigen::Vector3f& query, float minDistance) const;

private:
    cudex::DeviceSpan<const Point> points_;
    cudex::DeviceSpan<const Octant> octants_;
    cudex::DeviceSpan<const Id> successors_;
};


template<typename Point>
class Octree
{
public:
    void initialize(const std::vector<Point>& points);

    Id findNeighbor(const Point& query, float minDistance = -1) const;
    GPUQuery<Point> gpuQuery() const;

    ~Octree();

private:
    std::unique_ptr<impl::POctree<Point>> octree_;

    cudex::HostDeviceMemory<impl::Octant> octantsMem_;
    cudex::HostDeviceMemory<Id> successorsMem_;
    cudex::HostDeviceMemory<Point> pointsMem_;

    cudex::HostSpan<Point> hostPoints_;
};


// accessors for some types

template<size_t N>
__host__ __device__ float get(const float3& v);

template<size_t N>
__host__ __device__ float get(const Eigen::Vector3f& v);

}

#include "octree.inl.cu.h"
