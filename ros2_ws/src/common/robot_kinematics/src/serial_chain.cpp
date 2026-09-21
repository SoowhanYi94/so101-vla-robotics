#include "robot_kinematics/serial_chain.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace robot_kinematics
{

SerialChain::SerialChain(
    std::vector<Joint> joints,
    const Eigen::Isometry3d& tool_transform)
    : joints_(std::move(joints)),
      tool_transform_(tool_transform)
{
    if (joints_.empty())
    {
        throw std::invalid_argument("Serial chain cannot be empty.");
    }

    if (!tool_transform_.matrix().allFinite())
    {
        throw std::invalid_argument("Tool transform contains invalid values.");
    }

    for (Joint& joint : joints_)
    {
        if (joint.name.empty())
        {
            throw std::invalid_argument("Joint name cannot be empty.");
        }

        if (!joint.origin.matrix().allFinite() ||
            !joint.axis.allFinite())
        {
            throw std::invalid_argument(
                "Joint contains invalid values: " + joint.name);
        }

        if (joint.axis.norm() < 1.0e-12)
        {
            throw std::invalid_argument(
                "Joint axis cannot be zero: " + joint.name);
        }

        if (joint.lower_limit > joint.upper_limit)
        {
            throw std::invalid_argument(
                "Invalid joint limits: " + joint.name);
        }

        joint.axis.normalize();
    }
}

std::size_t SerialChain::size() const
{
    return joints_.size();
}

const std::vector<Joint>& SerialChain::joints() const
{
    return joints_;
}

Eigen::Isometry3d SerialChain::forwardKinematics(
    const Eigen::VectorXd& positions) const
{
    validateSize(positions);

    Eigen::Isometry3d transform =
        Eigen::Isometry3d::Identity();

    for (Eigen::Index index = 0;
         index < positions.size();
         ++index)
    {
        const Joint& joint =
            joints_.at(static_cast<std::size_t>(index));

        transform = transform * joint.origin;
        transform.rotate(
            Eigen::AngleAxisd(positions[index], joint.axis));
    }

    return transform * tool_transform_;
}

Eigen::MatrixXd SerialChain::jacobian(
    const Eigen::VectorXd& positions) const
{
    validateSize(positions);

    const Eigen::Index joint_count =
        static_cast<Eigen::Index>(joints_.size());

    std::vector<Eigen::Vector3d> joint_origins(
        joints_.size());

    std::vector<Eigen::Vector3d> joint_axes(
        joints_.size());

    Eigen::Isometry3d transform =
        Eigen::Isometry3d::Identity();

    for (Eigen::Index index = 0;
         index < joint_count;
         ++index)
    {
        const Joint& joint =
            joints_.at(static_cast<std::size_t>(index));

        transform = transform * joint.origin;

        joint_origins.at(static_cast<std::size_t>(index)) =
            transform.translation();

        joint_axes.at(static_cast<std::size_t>(index)) =
            transform.linear() * joint.axis;

        transform.rotate(
            Eigen::AngleAxisd(positions[index], joint.axis));
    }

    const Eigen::Vector3d end_effector_position =
        (transform * tool_transform_).translation();

    Eigen::MatrixXd result =
        Eigen::MatrixXd::Zero(6, joint_count);

    for (Eigen::Index index = 0;
         index < joint_count;
         ++index)
    {
        const Eigen::Vector3d& axis =
            joint_axes.at(static_cast<std::size_t>(index));

        const Eigen::Vector3d& origin =
            joint_origins.at(static_cast<std::size_t>(index));

        result.block<3, 1>(0, index) =
            axis.cross(end_effector_position - origin);

        result.block<3, 1>(3, index) =
            axis;
    }

    return result;
}

Eigen::VectorXd SerialChain::clampToLimits(
    const Eigen::VectorXd& positions) const
{
    validateSize(positions);

    Eigen::VectorXd result = positions;

    for (Eigen::Index index = 0;
         index < result.size();
         ++index)
    {
        const Joint& joint =
            joints_.at(static_cast<std::size_t>(index));

        result[index] = std::clamp(
            result[index],
            joint.lower_limit,
            joint.upper_limit);
    }

    return result;
}

bool SerialChain::isWithinLimits(
    const Eigen::VectorXd& positions) const
{
    validateSize(positions);

    for (Eigen::Index index = 0;
         index < positions.size();
         ++index)
    {
        const Joint& joint =
            joints_.at(static_cast<std::size_t>(index));

        if (positions[index] < joint.lower_limit ||
            positions[index] > joint.upper_limit)
        {
            return false;
        }
    }

    return true;
}

void SerialChain::validateSize(
    const Eigen::VectorXd& positions) const
{
    if (positions.size() !=
        static_cast<Eigen::Index>(joints_.size()))
    {
        throw std::invalid_argument(
            "Joint position vector has an incorrect size.");
    }

    if (!positions.allFinite())
    {
        throw std::invalid_argument(
            "Joint position vector contains invalid values.");
    }
}

}  // namespace robot_kinematics