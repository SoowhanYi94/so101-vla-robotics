#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "so101_control/robot_constants.hpp"

class SO101ActionController : public rclcpp::Node
{
public:
    SO101ActionController()
        : Node("so101_action_controller")
    {
        declare_parameters();
        read_parameters();
        validate_parameters();

        command_publisher_ = create_publisher<sensor_msgs::msg::JointState>(
            so101::JOINT_COMMAND_TOPIC, 10);

        state_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
            so101::JOINT_STATE_TOPIC, 10,
            std::bind(
                &SO101ActionController::joint_state_callback,
                this,
                std::placeholders::_1));

        policy_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
            so101::POLICY_ACTION_TOPIC, 10,
            std::bind(
                &SO101ActionController::policy_action_callback,
                this,
                std::placeholders::_1));

        const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / control_rate_hz_));

        control_timer_ = create_wall_timer(
            period, std::bind(&SO101ActionController::control_loop, this));

        RCLCPP_INFO(
            get_logger(),
            "SO-101 controller started in %s mode at %.1f Hz.",
            control_mode_name().c_str(),
            control_rate_hz_);

        RCLCPP_INFO(
            get_logger(),
            "Waiting for joint states on %s.",
            so101::JOINT_STATE_TOPIC);
    }

private:
    enum class ControlMode
    {
        Position,
        Velocity,
        Effort
    };

    ControlMode control_mode_ = ControlMode::Position;

    double control_rate_hz_ = 100.0;
    double state_timeout_seconds_ = 0.5;
    double action_timeout_seconds_ = 0.5;

    std::array<double, so101::NUM_JOINTS> maximum_velocities_{};
    std::array<double, so101::NUM_JOINTS> maximum_efforts_{};
    std::array<double, so101::NUM_JOINTS> current_positions_{};
    std::array<double, so101::NUM_JOINTS> requested_action_{};
    std::array<double, so101::NUM_JOINTS> last_position_command_{};

    bool received_state_ = false;
    bool received_action_ = false;
    bool timeout_warning_active_ = false;

    std::chrono::steady_clock::time_point last_state_time_;
    std::chrono::steady_clock::time_point last_action_time_;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr command_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr policy_subscription_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    void declare_parameters()
    {
        declare_parameter<std::string>("control_mode", "position");
        declare_parameter<double>("control_rate_hz", 100.0);
        declare_parameter<double>("state_timeout_seconds", 0.5);
        declare_parameter<double>("action_timeout_seconds", 0.5);

        declare_parameter<std::vector<double>>(
            "maximum_velocities",
            std::vector<double>(so101::NUM_JOINTS, 0.5));

        declare_parameter<std::vector<double>>(
            "maximum_efforts",
            std::vector<double>(so101::NUM_JOINTS, 0.2));
    }

    void read_parameters()
    {
        control_mode_ = parse_control_mode(
            get_parameter("control_mode").as_string());

        control_rate_hz_ = get_parameter("control_rate_hz").as_double();
        state_timeout_seconds_ = get_parameter("state_timeout_seconds").as_double();
        action_timeout_seconds_ = get_parameter("action_timeout_seconds").as_double();

        copy_parameter_array("maximum_velocities", maximum_velocities_);
        copy_parameter_array("maximum_efforts", maximum_efforts_);
    }

    void validate_parameters() const
    {
        if (control_rate_hz_ <= 0.0) {
            throw std::runtime_error("control_rate_hz must be positive");
        }

        if (state_timeout_seconds_ <= 0.0) {
            throw std::runtime_error("state_timeout_seconds must be positive");
        }

        if (action_timeout_seconds_ <= 0.0) {
            throw std::runtime_error("action_timeout_seconds must be positive");
        }

        for (std::size_t i = 0; i < so101::NUM_JOINTS; ++i) {
            if (maximum_velocities_[i] <= 0.0) {
                throw std::runtime_error("All maximum velocities must be positive");
            }

            if (maximum_efforts_[i] <= 0.0) {
                throw std::runtime_error("All maximum efforts must be positive");
            }
        }
    }

    void copy_parameter_array(
        const std::string &name,
        std::array<double, so101::NUM_JOINTS> &destination)
    {
        const auto values = get_parameter(name).as_double_array();

        if (values.size() != so101::NUM_JOINTS) {
            throw std::runtime_error(name + " must contain exactly six values");
        }

        std::copy(values.begin(), values.end(), destination.begin());
    }

    ControlMode parse_control_mode(const std::string &mode) const
    {
        if (mode == "position") {
            return ControlMode::Position;
        }

        if (mode == "velocity") {
            return ControlMode::Velocity;
        }

        if (mode == "effort") {
            return ControlMode::Effort;
        }

        throw std::runtime_error(
            "control_mode must be position, velocity, or effort");
    }

    std::string control_mode_name() const
    {
        switch (control_mode_) {
            case ControlMode::Position:
                return "position";
            case ControlMode::Velocity:
                return "velocity";
            case ControlMode::Effort:
                return "effort";
        }

        return "unknown";
    }

    bool reorder_values(
        const sensor_msgs::msg::JointState &message,
        const std::vector<double> &values,
        std::array<double, so101::NUM_JOINTS> &ordered) const
    {
        if (message.name.size() != values.size()) {
            return false;
        }

        std::unordered_map<std::string, double> received;

        for (std::size_t i = 0; i < message.name.size(); ++i) {
            received[message.name[i]] = values[i];
        }

        for (std::size_t i = 0; i < so101::NUM_JOINTS; ++i) {
            const auto result = received.find(so101::JOINT_NAMES[i]);

            if (result == received.end() || !std::isfinite(result->second)) {
                return false;
            }

            ordered[i] = result->second;
        }

        return true;
    }

    void joint_state_callback(
        const sensor_msgs::msg::JointState::SharedPtr message)
    {
        std::array<double, so101::NUM_JOINTS> positions{};

        if (!reorder_values(*message, message->position, positions)) {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Incomplete SO-101 position feedback.");
            return;
        }

        current_positions_ = positions;
        last_state_time_ = std::chrono::steady_clock::now();

        if (received_state_) {
            return;
        }

        received_state_ = true;
        last_position_command_ = current_positions_;

        RCLCPP_INFO(get_logger(), "All six SO-101 joints detected.");

        for (std::size_t i = 0; i < so101::NUM_JOINTS; ++i) {
            RCLCPP_INFO(
                get_logger(),
                "  %s: %.4f rad",
                so101::JOINT_NAMES[i].c_str(),
                current_positions_[i]);
        }

        RCLCPP_INFO(
            get_logger(),
            "Waiting for policy actions on %s.",
            so101::POLICY_ACTION_TOPIC);
    }

    const std::vector<double> &selected_action_field(
        const sensor_msgs::msg::JointState &message) const
    {
        switch (control_mode_) {
            case ControlMode::Position:
                return message.position;
            case ControlMode::Velocity:
                return message.velocity;
            case ControlMode::Effort:
                return message.effort;
        }

        return message.position;
    }

    void policy_action_callback(
        const sensor_msgs::msg::JointState::SharedPtr message)
    {
        std::array<double, so101::NUM_JOINTS> action{};
        const auto &values = selected_action_field(*message);

        if (!reorder_values(*message, values, action)) {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Rejected incomplete or invalid %s action.",
                control_mode_name().c_str());
            return;
        }

        clamp_action(action);

        requested_action_ = action;
        received_action_ = true;
        timeout_warning_active_ = false;
        last_action_time_ = std::chrono::steady_clock::now();
    }

    void clamp_action(std::array<double, so101::NUM_JOINTS> &action) const
    {
        for (std::size_t i = 0; i < so101::NUM_JOINTS; ++i) {
            switch (control_mode_) {
                case ControlMode::Position:
                    action[i] = std::clamp(
                        action[i],
                        so101::LOWER_POSITION_LIMITS[i],
                        so101::UPPER_POSITION_LIMITS[i]);
                    break;

                case ControlMode::Velocity:
                    action[i] = std::clamp(
                        action[i],
                        -maximum_velocities_[i],
                        maximum_velocities_[i]);
                    break;

                case ControlMode::Effort:
                    action[i] = std::clamp(
                        action[i],
                        -maximum_efforts_[i],
                        maximum_efforts_[i]);
                    break;
            }
        }
    }

    std::array<double, so101::NUM_JOINTS> fallback_action() const
    {
        if (control_mode_ == ControlMode::Position) {
            return last_position_command_;
        }

        // Zero velocity stops commanded movement.
        // Zero effort does not support the arm against gravity.
        return {};
    }

    bool state_is_fresh(
        const std::chrono::steady_clock::time_point &now) const
    {
        const double age = std::chrono::duration<double>(
            now - last_state_time_).count();

        return age <= state_timeout_seconds_;
    }

    bool action_is_fresh(
        const std::chrono::steady_clock::time_point &now) const
    {
        if (!received_action_) {
            return false;
        }

        const double age = std::chrono::duration<double>(
            now - last_action_time_).count();

        return age <= action_timeout_seconds_;
    }

    void populate_command(
        sensor_msgs::msg::JointState &command,
        const std::array<double, so101::NUM_JOINTS> &action)
    {
        switch (control_mode_) {
            case ControlMode::Position:
                command.position.assign(action.begin(), action.end());
                last_position_command_ = action;
                break;

            case ControlMode::Velocity:
                command.velocity.assign(action.begin(), action.end());
                break;

            case ControlMode::Effort:
                command.effort.assign(action.begin(), action.end());
                break;
        }
    }

    void control_loop()
    {
        if (!received_state_) {
            return;
        }

        const auto now = std::chrono::steady_clock::now();

        if (!state_is_fresh(now)) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "SO-101 feedback is stale. Command output is paused.");
            return;
        }

        std::array<double, so101::NUM_JOINTS> action{};

        if (action_is_fresh(now)) {
            action = requested_action_;
        } else {
            action = fallback_action();

            if (received_action_ && !timeout_warning_active_) {
                RCLCPP_WARN(
                    get_logger(),
                    "Policy action timed out. Applying the %s fallback.",
                    control_mode_name().c_str());

                timeout_warning_active_ = true;
            }
        }

        clamp_action(action);

        sensor_msgs::msg::JointState command;
        command.header.stamp = this->now();
        command.name.assign(
            so101::JOINT_NAMES.begin(),
            so101::JOINT_NAMES.end());

        populate_command(command, action);
        command_publisher_->publish(command);
    }
};


int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    try {
        rclcpp::spin(std::make_shared<SO101ActionController>());
    } catch (const std::exception &error) {
        RCLCPP_FATAL(
            rclcpp::get_logger("so101_action_controller"),
            "%s",
            error.what());
    }

    rclcpp::shutdown();
    return 0;
}