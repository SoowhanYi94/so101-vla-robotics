# ROS 2 Workspace

This ROS 2 Humble workspace contains reusable robotics packages under `src/common/` and
robot-specific packages under `src/robots/`.

- `robot_kinematics`: generic serial-chain FK, Jacobian, and IK.
- `spacemouse_teleop`: Cartesian teleoperation placeholder.
- `so101_description`: SO-101 model and kinematic configuration.
- `so101_control`: command validation, limits, timeouts, and control-mode output.
- `so101_bringup`: configuration and coordinated launch files.

## Build

```bash
cd "$(git rev-parse --show-toplevel)/ros2_ws"
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Generated `build/`, `install/`, and `log/` directories belong only here and should not be
committed. If packages move inside `src/`, remove those generated directories before rebuilding.

Useful diagnostics:

```bash
ros2 node list
ros2 topic list
ros2 topic echo /so101/joint_states --once
ros2 topic info /so101/end_effector_pose --verbose
ros2 node info /robot_kinematics
```