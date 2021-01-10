#pragma once

#include "exception.h"
#include "cudex/stack.cu.h"

#include "Octree.hpp"

constexpr inline size_t MAX_STACK_SIZE = 32;

constexpr static uint8_t INVALID_CHILD = 255;

using QPoint = Eigen::Vector3f;

namespace octree::impl {

// ----- GPU Octant

struct Octant
{
    QPoint point;
    float extent;

    uint32_t start;
    uint32_t size;

    bool isLeaf;

    Id children[8];

    __device__ bool containsBall(const QPoint& center, const float radius) const
    {
        assert(isfinite(radius));

        const Eigen::Array3f diff = (center.array() - point.array()).abs();
        return (diff + radius <= extent).all();
    }

    __device__ bool overlapsBall(const QPoint& center, const float radius) const
    {
        if (isinf(radius)) {
            return true;
        }

        assert(isfinite(radius));

        const Eigen::Array3f diff = (center - point).array().abs();
        const float maxDist = radius + extent;

        // If distance in any coordinate between centers is > than maxDist, there is no intersection
        if ((diff > maxDist).any()) {
            return false;
        }

        const size_t nLessExtent = (diff < extent).count();
        if (nLessExtent >= 2) {
            // ball center belongs to a projection of a quadrant on one of coordinate planes, and
            // the distance in the 3rd coordinate is less thant maxDist (checked in the previous step).

            return true;
        }

        const Eigen::Array3f diff2 = (diff - extent).max(0);
        assert((diff2 >=0).all());

        return diff2.matrix().squaredNorm() < radius * radius;
    }
};

// ----- Helper functions

struct StackState
{
    Id octantId = INVALID_ID;
    uint8_t bestChild = INVALID_CHILD;
    uint8_t currentChild = INVALID_CHILD;
};

template<size_t Dim, typename Point>
__device__ float getDim(const Point& point)
{
    using ::std::get;
    using ::octree::get;

    return get<Dim>(point);
}

template<typename Point>
__device__ Eigen::Vector3f toEigen(const Point& point)
{
    return {getDim<0>(point), getDim<1>(point), getDim<2>(point)};
}

// ----- POctree

template<typename Point>
class POctree: public unibn::Octree<Point, cudex::HostSpan<Point>>
{
public:
    using Base = unibn::Octree<Point, cudex::HostSpan<Point>>;
    using Octant = typename Base::Octant;

    void initialize(const cudex::HostSpan<Point>& points);

    const std::vector<uint32_t>& successors() const {
        return Base::successors_;
    };

    template<typename F>
    void walkOctants(const F& f) const;

    Id findNeighbor(const Point& point, float minDistance) const;

private:
    template<typename F>
    void walkOctantsImpl(const F& f, const Octant *, uint32_t) const;
};

template<typename Point>
void POctree<Point>::initialize(const cudex::HostSpan<Point>& points)
{
    const unibn::OctreeParams params(32, false, 0);
    Base::initialize(points, params);
}

template<typename Point>
template<typename F>
void POctree<Point>::walkOctants(const F& f) const
{
    walkOctantsImpl(f, Base::root_, 0);
}

template<typename Point>
template<typename F>
void POctree<Point>::walkOctantsImpl(const F& f, const Octant* octant, uint32_t level) const
{
    if (!octant) {
        return;
    }

    f(octant, level);
    for (const Octant* c : octant->child) {
        walkOctantsImpl(f, c, level + 1);
    }
}

template<typename Point>
Id POctree<Point>::findNeighbor(const Point& p, float minDistance) const
{
    return Base::template findNeighbor<unibn::L2Distance<Point>>(p, minDistance);
}

} // namespace octree::impl

namespace octree {

// ----- GPUQuery

template<typename Point>
__device__ GPUQuery<Point>::GPUQuery(
        cudex::DeviceSpan<const Point> points,
        cudex::DeviceSpan<const Octant> octants,
        cudex::DeviceSpan<const Id> successors)
    : points_(points)
    , octants_(octants)
    , successors_(successors)
{}

template<typename Point>
__device__ const Point& GPUQuery<Point>::point(Id index) const
{
    return points_[index];
}

template<typename Point>
__device__ size_t GPUQuery<Point>::nPoints() const
{
    return points_.size();
}

template<typename Point>
__device__ Id GPUQuery<Point>::findNeighbor(const Point& query, const float minDistance) const
{
    return findNeighborInternal(impl::toEigen(query), minDistance);
}

template<typename Point>
template<typename Derived>
__device__ Id GPUQuery<Point>::findNeighbor(const Eigen::MatrixBase<Derived>& query, float minDistance) const
{
    return findNeighborInternal(query, minDistance);
}

template<typename Point>
__device__ Id GPUQuery<Point>::findNeighborInternal(const QPoint& query, const float minDistance) const
{
    float maxDistance = INFINITY;
    float maxDistance2 = INFINITY;

    const float minDistance2 = minDistance < 0 ? minDistance : minDistance * minDistance;

    using StackState = impl::StackState;

    cudex::Stack<StackState, MAX_STACK_SIZE> stack;
    stack.push(StackState{0});

    Id closest = INVALID_ID;

    bool stop = false;
    while(!stack.empty() && !stop)
    {
        StackState& state = stack.top();

        assert(state.octantId != INVALID_ID);
        assert(state.bestChild < 8 || state.bestChild == INVALID_CHILD);

        if (state.currentChild == 8) {
            stack.pop();
            continue;
        }

        const Octant& octant = octants_[state.octantId];

        // Check leaf
        if (octant.isLeaf)
        {
            assert(octant.size > 0);

            Id currentId = octant.start;
            for (size_t i = 0; i < octant.size; ++i)
            {
                const float dist2 = (query - impl::toEigen(points_[currentId])).squaredNorm();
                if (minDistance2 < dist2 && dist2 < maxDistance2)
                {
                    maxDistance2 = dist2;
                    closest = currentId;
                }

                currentId = successors_[currentId];
            }

            maxDistance = sqrt(maxDistance2);
            stop = octant.containsBall(query, maxDistance);

            stack.pop();
            continue;
        }

        // Find most probable child
        if (state.currentChild == INVALID_CHILD) {
            assert(state.bestChild == INVALID_CHILD);

            state.currentChild = 0;
            uint8_t bestChild = 0;

            constexpr int DIMS[] = {0, 1, 2};

            for (auto dim : DIMS) {
                if (query[dim] > octant.point[dim]) {
                    bestChild |= (1 << dim);
                }
            }

            const Id childId = octant.children[bestChild];

            if (childId != INVALID_ID) {
                state.bestChild = bestChild;
                stack.push(StackState{childId});
                continue;
            }
        }

        assert(state.currentChild < 8);

        for (bool childPushed = false; state.currentChild < 8 && !childPushed; ++state.currentChild) {
            if (state.currentChild == state.bestChild) {
                continue;
            }

            const Id childId = octant.children[state.currentChild];
            if (childId == INVALID_ID) {
                continue;
            }

            const Octant& childOctant = octants_[childId];
            if (!childOctant.overlapsBall(query, maxDistance)) {
                continue;
            }

            stack.push(StackState{childId});
            childPushed = true;
        }
    }

    return closest;
}

// ----- Octree

template<typename Point>
void Octree<Point>::initialize(const std::vector<Point>& points)
{
    CHECK_GT(points.size(), 0);

    pointsMem_.resizeSync(points);
    hostPoints_ = pointsMem_.host();

    octree_ = std::make_unique<impl::POctree<Point>>();
    octree_->initialize(hostPoints_);
    
    uint32_t nodeCount = 0;
    uint32_t maxLevel = 0;

    using POctant = typename impl::POctree<Point>::Octant;

    std::unordered_map<const POctant*, uint32_t> map;

    octree_->walkOctants([&](const POctant* node, uint32_t level) {
        maxLevel = std::max(maxLevel, level);
        CHECK(map.find(node) == map.end());

        map[node] = nodeCount++;
    });

    octantsMem_.resize(nodeCount);
    CHECK_GT(nodeCount, 0);

    REQUIRE(maxLevel + 1 <= MAX_STACK_SIZE, "Too many tree levels: " << maxLevel);

    octree_->walkOctants([&](const POctant* node, uint32_t) {
        const Id id = map.at(node);
        impl::Octant& o = octantsMem_[id];
        o.point = QPoint(node->x, node->y, node->z);  
        o.extent = node->extent;
        o.start = node->start;
        o.size = node->size;
        o.isLeaf = node->isLeaf;

        for (size_t i = 0; i < 8; ++i) {
            o.children[i] = node->child[i] == nullptr ? INVALID_ID : map.at(node->child[i]);
            CHECK(o.children[i] != 0);
        }
    });

    octantsMem_.copyHostToDevice();

    successorsMem_.resizeSync(octree_->successors());
}

template<typename Point>
GPUQuery<Point> Octree<Point>::gpuQuery() const
{
    return { pointsMem_.device(), octantsMem_.device(), successorsMem_.device() };
}

template<typename Point>
Octree<Point>::~Octree() = default;

template<typename Point>
Id Octree<Point>::findNeighbor(const Point& point, float minDistance) const
{
    return octree_->findNeighbor(point, minDistance);
}

// ----- get accessors

template<size_t N>
__host__ __device__ float get(const float3& v)
{
    static_assert(N < 3);

    if constexpr(N == 0) {
        return v.x;
    }
    else if constexpr(N == 1) {
        return v.y;
    }

    return v.z;
}

template<size_t N>
__host__ __device__ float get(const Eigen::Vector3f& v)
{
    return v(N);
}


}
