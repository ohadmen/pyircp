#pragma once

#include <cstddef>
#include <vector>
#include <Eigen/Dense>

namespace alg {

struct ICPParams {
    size_t maxIterations = 10;
    size_t maxNonIncreaseIterations = 3;
    float  maxMatchingDistance = 1.0;

    size_t ransacIterations = 1000;
    float  ransacMaxInlierDistance = 0.1;

    int seed = 0;
};

struct ICPResult {
    enum Status {
        OK = 1,
        TOO_FEW_INPUT_POINTS,
        TOO_FEW_MATCHED_POINTS,
        RANSAC_FAILED
    };

    Status status = RANSAC_FAILED;

    Eigen::Matrix3f rotation;
    Eigen::Vector3f translation;

    size_t nInliers = 0;

    std::vector<size_t> inliers;    // Indexes of inliers

    bool valid() const {
        return status == OK;
    }
};

}

