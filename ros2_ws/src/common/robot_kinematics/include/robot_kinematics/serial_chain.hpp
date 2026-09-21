#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <string>
#include <vector>

namespace robot_kinematics
{

struct Joint
{
    std::string name;
    std::string parent_link;
    std::string child_link;

    Eigen::Isometry3d origin = Eigen::Isometry3d::Identity();
    Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();

    double lower_limit = 0.0;
    double upper_limit = 0.0;
};

class SerialChain
{
public:
    SerialChain(
        std::vector<Joint> joints,
        const Eigen::Isometry3d& tool_transform);

    std::size_t size() const;

    const std::vector<Joint>& joints() const;

    Eigen::Isometry3d forwardKinematics(
        const Eigen::VectorXd& positions) const;

    Eigen::MatrixXd jacobian(
        const Eigen::VectorXd& positions) const;

    Eigen::VectorXd clampToLimits(
        const Eigen::VectorXd& positions) const;

    bool isWithinLimits(
        const Eigen::VectorXd& positions) const;

private:
    void validateSize(
        const Eigen::VectorXd& positions) const;

    std::vector<Joint> joints_;
    Eigen::Isometry3d tool_transform_;
};

}  // namespace robot_kinematics