cmake_minimum_required(VERSION 3.8)
project(robot_kinematics)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(robot_kinematics_core
  src/serial_chain.cpp
  src/model_loader.cpp
  src/ik_solver.cpp
)

target_compile_features(robot_kinematics_core PUBLIC cxx_std_17)

target_include_directories(robot_kinematics_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(robot_kinematics_core
  PUBLIC
    Eigen3::Eigen
    yaml-cpp
)

add_executable(kinematics_node
  src/kinematics_node.cpp
)

target_compile_features(kinematics_node PUBLIC cxx_std_17)

target_link_libraries(kinematics_node
  robot_kinematics_core
)

ament_target_dependencies(kinematics_node
  geometry_msgs
  rclcpp
  sensor_msgs
  std_msgs
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS robot_kinematics_core
  EXPORT export_robot_kinematics
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS kinematics_node
  DESTINATION lib/${PROJECT_NAME}
)


ament_export_targets(
  export_robot_kinematics
  HAS_LIBRARY_TARGET
)

ament_export_dependencies(
  Eigen3
  yaml-cpp
)

ament_package()