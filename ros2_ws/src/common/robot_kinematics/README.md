# Robot Kinematics

`robot_kinematics` is a robot-independent C++ ROS 2 package. Robot geometry is loaded from YAML,
allowing the same serial-chain implementation to support future arms.

- `serial_chain`: forward kinematics, geometric Jacobian, and joint limits.
- `model_loader`: YAML origins, axes, limits, and tool transform.
- `ik_solver`: damped least-squares IK with bounded steps and seed continuity.
- `kinematics_node`: joint-state input, tool-pose publication, Cartesian command input, and joint
  action output.

The SO-101 arm chain contains five revolute joints. Its gripper is passed through as the sixth
controller command and is not part of arm FK or IK. Since the arm has five pose degrees of
freedom, position is the primary IK objective and orientation is weighted more softly.

Interfaces:

| Topic | Direction | Purpose |
|---|---|---|
| `/so101/joint_states` | Input | Current joint measurements |
| `/so101/end_effector_pose` | Output | FK-computed tool pose |
| `/so101/cartesian_command` | Input | Desired IK pose |
| `/so101/policy_action` | Output | Solved joint target |

Validate FK against the Isaac Sim tool transform before relying on Cartesian control.