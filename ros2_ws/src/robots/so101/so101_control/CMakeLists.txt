cmake_minimum_required(VERSION 3.8)

project(so101_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(
  action_controller
  src/action_controller_node.cpp
)

target_include_directories(
  action_controller
  PRIVATE include
)

ament_target_dependencies(
  action_controller
  rclcpp
  sensor_msgs
)

install(
  TARGETS action_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()