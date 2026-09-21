#include "robot_kinematics/ik_solver.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace robot_kinematics
{

IkSolver::IkSolver(
    const SerialChain& chain,
    IkOptions options)
    : chain_(chain),
      options_(options)
{
    if (options_.maximum_iterations == 0)
    {
        throw std::invalid_argument(
            "maximum_iterations must be greater than zero.");
    }

    if (options_.position_tolerance <= 0.0 ||
        options_.orientation_tolerance <= 0.0)
    {
        throw std::invalid_argument(
            "IK tolerances must be positive.");
    }

    if (options_.position_weight < 0.0 ||
        options_.orientation_weight < 0.0)
    {
        throw std::invalid_argument(
            "IK weights cannot be negative.");
    }

    if (options_.damping <= 0.0 ||
        options_.continuity_weight < 0.0 ||
        options_.maximum_joint_step <= 0.0)
    {
        throw std::invalid_argument(
            "Invalid IK solver option.");
    }
}

IkResult IkSolver::solve(
    const Eigen::Isometry3d& target,
    const Eigen::VectorXd& initial_positions) const
{
    if (!target.matrix().allFinite())
    {
        throw std::invalid_argument(
            "IK target contains invalid values.");
    }

    Eigen::VectorXd positions =
        chain_.clampToLimits(initial_positions);

    IkResult result;
    result.positions = positions;

    for (std::size_t iteration = 0;
         iteration < options_.maximum_iterations;
         ++iteration)
    {
        const Eigen::Isometry3d current =
            chain_.forwardKinematics(positions);

        const Eigen::Matrix<double, 6, 1> error =
            calculateError(current, target);

        result.position_error =
            error.head<3>().norm();

        result.orientation_error =
            error.tail<3>().norm();

        const bool position_converged =
            result.position_error <=
            options_.position_tolerance;

        const bool orientation_converged =
            options_.orientation_weight == 0.0 ||
            result.orientation_error <=
            options_.orientation_tolerance;

        if (position_converged && orientation_converged)
        {
            result.success = true;
            result.positions = positions;
            result.iterations = iteration;
            return result;
        }

        Eigen::MatrixXd weighted_jacobian =
            chain_.jacobian(positions);

        Eigen::Matrix<double, 6, 1> weighted_error =
            error;

        weighted_jacobian.topRows(3) *=
            options_.position_weight;

        weighted_jacobian.bottomRows(3) *=
            options_.orientation_weight;

        weighted_error.head<3>() *=
            options_.position_weight;

        weighted_error.tail<3>() *=
            options_.orientation_weight;

        const Eigen::Index joint_count =
            positions.size();

        Eigen::MatrixXd system =
            weighted_jacobian.transpose() *
            weighted_jacobian;

        system.diagonal().array() +=
            options_.damping * options_.damping +
            options_.continuity_weight;

        Eigen::VectorXd right_hand_side =
            weighted_jacobian.transpose() *
            weighted_error;

        right_hand_side +=
            options_.continuity_weight *
            (initial_positions - positions);

        Eigen::LDLT<Eigen::MatrixXd> decomposition(
            system);

        if (decomposition.info() != Eigen::Success)
        {
            break;
        }

        Eigen::VectorXd joint_step =
            decomposition.solve(right_hand_side);

        if (!joint_step.allFinite())
        {
            break;
        }

        const double largest_step =
            joint_step.cwiseAbs().maxCoeff();

        if (largest_step > options_.maximum_joint_step)
        {
            joint_step *=
                options_.maximum_joint_step /
                largest_step;
        }

        positions =
            chain_.clampToLimits(
                positions + joint_step);

        result.positions = positions;
        result.iterations = iteration + 1;
    }

    const Eigen::Isometry3d final_pose =
        chain_.forwardKinematics(result.positions);

    const Eigen::Matrix<double, 6, 1> final_error =
        calculateError(final_pose, target);

    result.position_error =
        final_error.head<3>().norm();

    result.orientation_error =
        final_error.tail<3>().norm();

    return result;
}

Eigen::Matrix<double, 6, 1> IkSolver::calculateError(
    const Eigen::Isometry3d& current,
    const Eigen::Isometry3d& target) const
{
    Eigen::Matrix<double, 6, 1> error;

    error.head<3>() =
        target.translation() -
        current.translation();

    const Eigen::Matrix3d rotation_error =
        target.linear() *
        current.linear().transpose();

    const Eigen::AngleAxisd angle_axis(
        rotation_error);

    if (std::abs(angle_axis.angle()) < 1.0e-12)
    {
        error.tail<3>().setZero();
    }
    else
    {
        error.tail<3>() =
            angle_axis.axis() *
            angle_axis.angle();
    }

    return error;
}

}  // namespace robot_kinematics