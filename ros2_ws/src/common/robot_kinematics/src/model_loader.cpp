#include "robot_kinematics/model_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>

#include <stdexcept>
#include <string>
#include <vector>

namespace robot_kinematics
{
namespace
{

Eigen::Vector3d readVector3(
    const YAML::Node& node,
    const std::string& field_name)
{
    if (!node || !node.IsSequence() || node.size() != 3)
    {
        throw std::runtime_error(
            field_name + " must contain exactly three values.");
    }

    return {
        node[0].as<double>(),
        node[1].as<double>(),
        node[2].as<double>()
    };
}

Eigen::Isometry3d readTransform(
    const YAML::Node& origin,
    const std::string& field_name)
{
    if (!origin)
    {
        throw std::runtime_error(
            "Missing transform: " + field_name);
    }

    const Eigen::Vector3d xyz =
        readVector3(origin["xyz"], field_name + ".xyz");

    const Eigen::Vector3d rpy =
        readVector3(origin["rpy"], field_name + ".rpy");

    const Eigen::AngleAxisd roll(
        rpy.x(),
        Eigen::Vector3d::UnitX());

    const Eigen::AngleAxisd pitch(
        rpy.y(),
        Eigen::Vector3d::UnitY());

    const Eigen::AngleAxisd yaw(
        rpy.z(),
        Eigen::Vector3d::UnitZ());

    Eigen::Isometry3d transform =
        Eigen::Isometry3d::Identity();

    transform.translation() = xyz;
    transform.linear() =
        (yaw * pitch * roll).toRotationMatrix();

    return transform;
}

Joint readJoint(
    const YAML::Node& joints,
    const std::string& joint_name)
{
    const YAML::Node node = joints[joint_name];

    if (!node)
    {
        throw std::runtime_error(
            "Missing joint configuration: " + joint_name);
    }

    const std::string type =
        node["type"].as<std::string>();

    if (type != "revolute")
    {
        throw std::runtime_error(
            "Unsupported joint type for " + joint_name +
            ": " + type);
    }

    Joint joint;

    joint.name = joint_name;
    joint.parent_link =
        node["parent"].as<std::string>();
    joint.child_link =
        node["child"].as<std::string>();

    joint.origin = readTransform(
        node["origin"],
        "joints." + joint_name + ".origin");

    joint.axis = readVector3(
        node["axis"],
        "joints." + joint_name + ".axis");

    joint.lower_limit =
        node["limits"]["lower"].as<double>();

    joint.upper_limit =
        node["limits"]["upper"].as<double>();

    return joint;
}

}  // namespace

SerialChain loadSerialChain(
    const std::string& configuration_path)
{
    try
    {
        const YAML::Node root =
            YAML::LoadFile(configuration_path);

        const YAML::Node robot = root["robot"];

        if (!robot)
        {
            throw std::runtime_error(
                "Missing robot section in kinematics configuration.");
        }

        const YAML::Node joint_names =
            robot["arm_joint_names"];

        const YAML::Node joints =
            robot["joints"];

        if (!joint_names ||
            !joint_names.IsSequence() ||
            joint_names.size() == 0)
        {
            throw std::runtime_error(
                "arm_joint_names must contain at least one joint.");
        }

        if (!joints || !joints.IsMap())
        {
            throw std::runtime_error(
                "Missing joints section.");
        }

        std::vector<Joint> chain_joints;
        chain_joints.reserve(joint_names.size());

        for (const YAML::Node& joint_name_node : joint_names)
        {
            const std::string joint_name =
                joint_name_node.as<std::string>();

            chain_joints.push_back(
                readJoint(joints, joint_name));
        }

        const Eigen::Isometry3d tool_transform =
            readTransform(
                robot["end_effector"]["origin"],
                "end_effector.origin");

        return SerialChain(
            std::move(chain_joints),
            tool_transform);
    }
    catch (const YAML::Exception& error)
    {
        throw std::runtime_error(
            "Failed to load kinematics configuration '" +
            configuration_path + "': " + error.what());
    }
}

}  // namespace robot_kinematics