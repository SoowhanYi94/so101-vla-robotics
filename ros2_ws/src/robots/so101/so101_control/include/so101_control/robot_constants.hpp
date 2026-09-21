#ifndef SO101_CONTROL__ROBOT_CONSTANTS_HPP_
#define SO101_CONTROL__ROBOT_CONSTANTS_HPP_

#include <array>
#include <cstddef>
#include <string>

namespace so101
{

inline constexpr std::size_t NUM_JOINTS = 6;

inline const std::array<std::string, NUM_JOINTS>
JOINT_NAMES = {
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
};

inline constexpr std::array<double, NUM_JOINTS>
LOWER_POSITION_LIMITS = {
    -1.919862,
    -1.745329,
    -1.690000,
    -1.658063,
    -2.743847,
    -0.174533,
};

inline constexpr std::array<double, NUM_JOINTS>
UPPER_POSITION_LIMITS = {
    1.919862,
    1.745329,
    1.690000,
    1.658063,
    2.841206,
    1.745329,
};

inline constexpr char JOINT_STATE_TOPIC[] =
    "/so101/joint_states";

inline constexpr char JOINT_COMMAND_TOPIC[] =
    "/so101/joint_commands";

inline constexpr char POLICY_ACTION_TOPIC[] =
    "/so101/policy_action";

}  // namespace so101

#endif  // SO101_CONTROL__ROBOT_CONSTANTS_HPP_