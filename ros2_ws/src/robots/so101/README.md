# SO-101 ROS 2 Packages

The SO-101 integration is split by responsibility:

- `so101_description`: joint names, limits, transforms, and kinematic YAML.
- `so101_control`: low-level command contract, validation, limits, and timeouts.
- `so101_bringup`: launch files and runtime parameter composition.

Controller joint order is:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

Position and velocity commands use the simulator or motor drive's closed loop. Effort mode sends
torque commands and requires appropriate stiffness, damping, gravity handling, and conservative
effort limits. Use only one primary command mode per joint in a control cycle.

```bash
ros2 launch so101_control controller.launch.py
ros2 launch so101_bringup kinematics.launch.py
ros2 launch so101_bringup vla.launch.py \
  checkpoint_path:=/absolute/path/to/best.pt
```

Downloaded actions must pass through `so101_control`; the learned policy must never bypass the
deterministic limit and timeout checks.