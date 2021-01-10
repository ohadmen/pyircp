#include "icp_wrapper.h"
#include "icp.cu.h"


ICPWrapper::ICPWrapper(const alg::ICPParams& params)
{
    icp_ = std::make_unique<alg::ICP>(params);
}

ICPWrapper::~ICPWrapper() = default;

alg::ICPResult ICPWrapper::run(
        const EigenPoints& src,
        const EigenPoints& dst,
        const EigenPoints& dstNormals,
        const Eigen::Matrix3f& initialRotation,
        const Eigen::Vector3f& initialTranslation)
{
    auto toSpan = [](const auto& matrix) {
        return cudex::HostSpan<const float3>(
            reinterpret_cast<const float3*>(matrix.data()),
            matrix.rows()
        );
    };

    return icp_->run(toSpan(src), toSpan(dst), toSpan(dstNormals), initialRotation, initialTranslation);
}

