#pragma once

#include "robot_kinematics/serial_chain.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>

namespace robot_kinematics
{

struct IkOptions
{
    std::size_t maximum_iterations = 100;

    double position_tolerance = 1.0e-4;
    double orientation_tolerance = 1.0e-2;

    double position_weight = 1.0;
    double orientation_weight = 0.15;

    double damping = 1.0e-3;
    double continuity_weight = 1.0e-3;
    double maximum_joint_step = 0.10;
};

struct IkResult
{
    bool success = false;
    Eigen::VectorXd positions;

    std::size_t iterations = 0;

    double position_error = 0.0;
    double orientation_error = 0.0;
};

class IkSolver
{
public:
    IkSolver(
        const SerialChain& chain,
        IkOptions options = {});

    IkResult solve(
        const Eigen::Isometry3d& target,
        const Eigen::VectorXd& initial_positions) const;

private:
    Eigen::Matrix<double, 6, 1> calculateError(
        const Eigen::Isometry3d& current,
        const Eigen::Isometry3d& target) const;

    const SerialChain& chain_;
    IkOptions options_;
};

}  // namespace robot_kinematics