# Isaac Sim Integration

The simulation layer loads the SO-101 USD, builds the task scene, and bridges robot observations
and commands to ROS 2.

| File | Responsibility |
|---|---|
| `isaac_sim/run_so101_sim.py` | Isaac Sim application and simulation loop |
| `isaac_sim/so101_scene.py` | Robot, table, lighting, objects, and camera |
| `isaac_sim/ros2_joint_bridge.py` | Joint states and command application |
| `isaac_sim/ros2_camera_bridge.py` | RGB image and camera information |

Run with Isaac Sim's Python interpreter:

```bash
cd "$(git rev-parse --show-toplevel)/simulation/isaac_sim"
/path/to/isaac-sim/python.sh run_so101_sim.py
```

Keep simulation-only APIs in this directory. ROS message contracts should remain stable so the
same policy and controller can later target physical hardware. Validate joint ordering, units,
camera topic, physics rate, and articulation drive settings after changing the USD or scene.