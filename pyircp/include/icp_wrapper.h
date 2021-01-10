#pragma once

#include "icp_types.h"

#include "Eigen/Dense"

#include <memory>
#include <vector>

namespace alg {
class ICP;
}

class ICPWrapper
{
public:
    using EigenPoints = Eigen::Ref<
        const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>,
        Eigen::Unaligned,
        Eigen::Stride<3, 1>
    >;

    ICPWrapper(const alg::ICPParams& params);
    ~ICPWrapper();

    alg::ICPResult run(
        const EigenPoints& src,
        const EigenPoints& dst,
        const EigenPoints& dstNormals,
        const Eigen::Matrix3f& initialRotation,
        const Eigen::Vector3f& initialTranslation);


private:
    std::unique_ptr<alg::ICP> icp_;
};

