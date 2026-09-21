import time
from importlib.resources import files

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState

from so101_policies.common.trajectory import interpolate_joint_positions
from so101_policies.constants import (
    JOINT_NAMES,
    JOINT_STATE_TOPIC,
    LOWER_POSITION_LIMITS,
    POLICY_ACTION_TOPIC,
    UPPER_POSITION_LIMITS,
)


class PoseSequencePolicy(Node):
    def __init__(self):
        super().__init__("so101_pose_sequence_policy")

        self.config = self.load_config()
        self.validate_config()

        self.transition_duration = float(self.config["transition_duration_seconds"])
        self.hold_duration = float(self.config["hold_duration_seconds"])
        self.publish_rate = float(self.config["publish_rate_hz"])
        self.relative_to_initial = bool(
            self.config.get("relative_to_initial", True)
        )

        self.pose_definitions = self.config["poses"]
        self.sequence = self.config["sequence"]

        self.initial_positions = None
        self.start_positions = None
        self.target_positions = None
        self.final_positions = None

        self.sequence_index = 0
        self.phase = "waiting"
        self.phase_start_time = None
        self.completion_logged = False

        self.publisher = self.create_publisher(
            JointState,
            POLICY_ACTION_TOPIC,
            10,
        )

        self.subscription = self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self.joint_state_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.control_loop,
        )

        self.get_logger().info(
            f"Waiting for joint states on {JOINT_STATE_TOPIC}."
        )

    def load_config(self):
        config_path = files("so101_policies").joinpath(
            "config",
            "poses.yaml",
        )

        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def validate_config(self):
        required_fields = {
            "transition_duration_seconds",
            "hold_duration_seconds",
            "publish_rate_hz",
            "poses",
            "sequence",
        }

        missing_fields = required_fields - self.config.keys()

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Missing configuration fields: {missing}")

        if self.config["transition_duration_seconds"] <= 0.0:
            raise ValueError("transition_duration_seconds must be positive")

        if self.config["hold_duration_seconds"] < 0.0:
            raise ValueError("hold_duration_seconds cannot be negative")

        if self.config["publish_rate_hz"] <= 0.0:
            raise ValueError("publish_rate_hz must be positive")

        if not self.config["sequence"]:
            raise ValueError("The pose sequence cannot be empty")

        for pose_name in self.config["sequence"]:
            if pose_name not in self.config["poses"]:
                raise ValueError(f"Unknown pose in sequence: {pose_name}")

            pose = self.config["poses"][pose_name]
            missing_joints = set(JOINT_NAMES) - pose.keys()

            if missing_joints:
                missing = ", ".join(sorted(missing_joints))
                raise ValueError(
                    f"Pose '{pose_name}' is missing joints: {missing}"
                )

    def joint_state_callback(self, message):
        if self.initial_positions is not None:
            return

        if len(message.name) != len(message.position):
            return

        received = dict(zip(message.name, message.position))

        if not all(name in received for name in JOINT_NAMES):
            return

        self.initial_positions = np.array(
            [received[name] for name in JOINT_NAMES],
            dtype=np.float64,
        )

        self.start_positions = self.initial_positions.copy()
        self.begin_pose(0)

        self.get_logger().info(
            "Initial joint state received. Starting pose sequence."
        )

    def pose_to_array(self, pose_name):
        pose = self.pose_definitions[pose_name]

        values = np.array(
            [pose[name] for name in JOINT_NAMES],
            dtype=np.float64,
        )

        if self.relative_to_initial:
            values += self.initial_positions

        return self.clamp_positions(values)

    def clamp_positions(self, positions):
        return np.clip(
            positions,
            np.array(LOWER_POSITION_LIMITS),
            np.array(UPPER_POSITION_LIMITS),
        )

    def begin_pose(self, sequence_index):
        self.sequence_index = sequence_index
        pose_name = self.sequence[sequence_index]

        self.target_positions = self.pose_to_array(pose_name)
        self.phase = "transition"
        self.phase_start_time = time.monotonic()

        self.get_logger().info(f"Moving to pose: {pose_name}")

    def control_loop(self):
        if self.initial_positions is None:
            return

        now = time.monotonic()
        elapsed = now - self.phase_start_time

        if self.phase == "transition":
            self.update_transition(elapsed)
        elif self.phase == "hold":
            self.update_hold(elapsed)
        elif self.phase == "complete":
            self.publish_positions(self.final_positions)

    def update_transition(self, elapsed):
        progress = min(elapsed / self.transition_duration, 1.0)

        positions = interpolate_joint_positions(
            self.start_positions,
            self.target_positions,
            progress,
        )

        self.publish_positions(positions)

        if progress >= 1.0:
            self.final_positions = self.target_positions.copy()
            self.phase = "hold"
            self.phase_start_time = time.monotonic()

    def update_hold(self, elapsed):
        self.publish_positions(self.target_positions)

        if elapsed < self.hold_duration:
            return

        next_index = self.sequence_index + 1

        if next_index >= len(self.sequence):
            self.phase = "complete"
            self.final_positions = self.target_positions.copy()

            if not self.completion_logged:
                self.get_logger().info(
                    "Pose sequence completed. Holding final pose."
                )
                self.completion_logged = True

            return

        self.start_positions = self.target_positions.copy()
        self.begin_pose(next_index)

    def publish_positions(self, positions):
        positions = self.clamp_positions(positions)

        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = JOINT_NAMES.copy()
        message.position = positions.tolist()

        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = PoseSequencePolicy()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Pose sequence stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()