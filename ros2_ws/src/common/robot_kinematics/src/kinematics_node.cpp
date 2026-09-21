#include "robot_kinematics/ik_solver.hpp"
#include "robot_kinematics/model_loader.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace robot_kinematics
{

class KinematicsNode : public rclcpp::Node
{
public:
    KinematicsNode()
        : Node("robot_kinematics"),
          configuration_path_(
              declare_parameter<std::string>(
                  "kinematics_config")),
          chain_(loadSerialChain(configuration_path_)),
          solver_(chain_, loadOptions()),
          arm_positions_(
              Eigen::VectorXd::Zero(
                  static_cast<Eigen::Index>(
                      chain_.size())))
    {
        base_frame_ =
            declare_parameter<std::string>(
                "base_frame",
                "base_link");

        gripper_joint_name_ =
            declare_parameter<std::string>(
                "gripper_joint_name",
                "gripper");

        joint_state_topic_ =
            declare_parameter<std::string>(
                "joint_state_topic",
                "/so101/joint_states");

        cartesian_command_topic_ =
            declare_parameter<std::string>(
                "cartesian_command_topic",
                "/so101/cartesian_command");

        policy_action_topic_ =
            declare_parameter<std::string>(
                "policy_action_topic",
                "/so101/policy_action");
        
        end_effector_pose_topic_ =
            declare_parameter<std::string>(
                "end_effector_pose_topic",
                "/so101/end_effector_pose");
        
        

        joint_state_subscription_ =
            create_subscription<sensor_msgs::msg::JointState>(
                joint_state_topic_,
                rclcpp::SensorDataQoS(),
                std::bind(
                    &KinematicsNode::handleJointState,
                    this,
                    std::placeholders::_1));

        cartesian_command_subscription_ =
            create_subscription<
                geometry_msgs::msg::PoseStamped>(
                cartesian_command_topic_,
                10,
                std::bind(
                    &KinematicsNode::handleCartesianCommand,
                    this,
                    std::placeholders::_1));

        policy_action_publisher_ =
            create_publisher<
                std_msgs::msg::Float64MultiArray>(
                policy_action_topic_,
                10);
        
        end_effector_pose_publisher_ =
            create_publisher<geometry_msgs::msg::PoseStamped>(
                end_effector_pose_topic_,
                10);
        
        RCLCPP_INFO(
            get_logger(),
            "Loaded %zu-joint chain from %s",
            chain_.size(),
            configuration_path_.c_str());
    }

private:
    IkOptions loadOptions()
    {
        IkOptions options;

        options.maximum_iterations =
            static_cast<std::size_t>(
                declare_parameter<int>(
                    "ik.maximum_iterations",
                    100));

        options.position_tolerance =
            declare_parameter<double>(
                "ik.position_tolerance",
                1.0e-4);

        options.orientation_tolerance =
            declare_parameter<double>(
                "ik.orientation_tolerance",
                1.0e-2);

        options.position_weight =
            declare_parameter<double>(
                "ik.position_weight",
                1.0);

        options.orientation_weight =
            declare_parameter<double>(
                "ik.orientation_weight",
                0.15);

        options.damping =
            declare_parameter<double>(
                "ik.damping",
                1.0e-3);

        options.continuity_weight =
            declare_parameter<double>(
                "ik.continuity_weight",
                1.0e-3);

        options.maximum_joint_step =
            declare_parameter<double>(
                "ik.maximum_joint_step",
                0.10);

        return options;
    }

    void handleJointState(
        const sensor_msgs::msg::JointState::SharedPtr message)
    {
        Eigen::VectorXd positions =
            Eigen::VectorXd::Zero(
                static_cast<Eigen::Index>(
                    chain_.size()));

        bool complete = true;

        for (std::size_t joint_index = 0;
             joint_index < chain_.size();
             ++joint_index)
        {
            const std::string& joint_name =
                chain_.joints()[joint_index].name;

            const auto iterator =
                std::find(
                    message->name.begin(),
                    message->name.end(),
                    joint_name);

            if (iterator == message->name.end())
            {
                complete = false;
                break;
            }

            const std::size_t message_index =
                static_cast<std::size_t>(
                    std::distance(
                        message->name.begin(),
                        iterator));

            if (message_index >=
                message->position.size())
            {
                complete = false;
                break;
            }

            positions[
                static_cast<Eigen::Index>(
                    joint_index)] =
                message->position[message_index];
        }

        if (!complete || !positions.allFinite())
        {
            return;
        }

        double gripper_position =
            gripper_position_;

        const auto gripper_iterator =
            std::find(
                message->name.begin(),
                message->name.end(),
                gripper_joint_name_);

        if (gripper_iterator != message->name.end())
        {
            const std::size_t gripper_index =
                static_cast<std::size_t>(
                    std::distance(
                        message->name.begin(),
                        gripper_iterator));

            if (gripper_index <
                message->position.size())
            {
                gripper_position =
                    message->position[gripper_index];
            }
        }

        {
        std::lock_guard<std::mutex> lock(state_mutex_);

        arm_positions_ = positions;
        gripper_position_ = gripper_position;
        has_joint_state_ = true;
    }
    const Eigen::Isometry3d end_effector_pose =
        chain_.forwardKinematics(positions);

    const Eigen::Quaterniond orientation(
        end_effector_pose.linear());

    geometry_msgs::msg::PoseStamped pose_message;

    pose_message.header = message->header;
    pose_message.header.frame_id = base_frame_;

    pose_message.pose.position.x =
        end_effector_pose.translation().x();

    pose_message.pose.position.y =
        end_effector_pose.translation().y();

    pose_message.pose.position.z =
        end_effector_pose.translation().z();

    pose_message.pose.orientation.x = orientation.x();
    pose_message.pose.orientation.y = orientation.y();
    pose_message.pose.orientation.z = orientation.z();
    pose_message.pose.orientation.w = orientation.w();

    end_effector_pose_publisher_->publish(
        pose_message);
    }

    void handleCartesianCommand(
        const geometry_msgs::msg::PoseStamped::SharedPtr message)
    {
        if (!message->header.frame_id.empty() &&
            message->header.frame_id != base_frame_)
        {
            RCLCPP_WARN(
                get_logger(),
                "Rejected target in frame '%s'; expected '%s'.",
                message->header.frame_id.c_str(),
                base_frame_.c_str());

            return;
        }

        Eigen::VectorXd initial_positions;
        double gripper_position = 0.0;

        {
            std::lock_guard<std::mutex> lock(
                state_mutex_);

            if (!has_joint_state_)
            {
                RCLCPP_WARN(
                    get_logger(),
                    "Waiting for a complete joint state.");

                return;
            }

            initial_positions = arm_positions_;
            gripper_position = gripper_position_;
        }

        const auto& pose = message->pose;

        Eigen::Quaterniond orientation(
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z);

        if (!orientation.coeffs().allFinite() ||
            orientation.norm() < 1.0e-12)
        {
            RCLCPP_WARN(
                get_logger(),
                "Rejected invalid target orientation.");

            return;
        }

        orientation.normalize();

        Eigen::Isometry3d target =
            Eigen::Isometry3d::Identity();

        target.translation() = Eigen::Vector3d(
            pose.position.x,
            pose.position.y,
            pose.position.z);

        target.linear() =
            orientation.toRotationMatrix();

        if (!target.matrix().allFinite())
        {
            RCLCPP_WARN(
                get_logger(),
                "Rejected non-finite Cartesian target.");

            return;
        }

        const IkResult result =
            solver_.solve(
                target,
                initial_positions);

        if (!result.success)
        {
            RCLCPP_WARN(
                get_logger(),
                "IK did not converge: position error %.6f m, "
                "orientation error %.6f rad.",
                result.position_error,
                result.orientation_error);

            return;
        }

        std_msgs::msg::Float64MultiArray command;

        command.data.reserve(
            chain_.size() + 1);

        for (Eigen::Index index = 0;
             index < result.positions.size();
             ++index)
        {
            command.data.push_back(
                result.positions[index]);
        }

        command.data.push_back(
            gripper_position);

        policy_action_publisher_->publish(
            command);
    }

    std::string configuration_path_;
    SerialChain chain_;
    IkSolver solver_;

    std::string base_frame_;
    std::string gripper_joint_name_;
    std::string joint_state_topic_;
    std::string cartesian_command_topic_;
    std::string policy_action_topic_;
    std::string end_effector_pose_topic_;

    std::mutex state_mutex_;
    Eigen::VectorXd arm_positions_;
    double gripper_position_ = 0.0;
    bool has_joint_state_ = false;

    rclcpp::Subscription<
        sensor_msgs::msg::JointState>::SharedPtr
        joint_state_subscription_;

    rclcpp::Subscription<
        geometry_msgs::msg::PoseStamped>::SharedPtr
        cartesian_command_subscription_;

    rclcpp::Publisher<
        std_msgs::msg::Float64MultiArray>::SharedPtr
        policy_action_publisher_;

    rclcpp::Publisher<
        geometry_msgs::msg::PoseStamped>::SharedPtr
        end_effector_pose_publisher_;
};

}  // namespace robot_kinematics

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::spin(
        std::make_shared<
            robot_kinematics::KinematicsNode>());

    rclcpp::shutdown();

    return 0;
}