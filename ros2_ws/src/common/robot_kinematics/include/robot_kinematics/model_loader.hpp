#pragma once

#include "robot_kinematics/serial_chain.hpp"

#include <string>

namespace robot_kinematics
{

SerialChain loadSerialChain(
    const std::string& configuration_path);

}  // namespace robot_kinematics