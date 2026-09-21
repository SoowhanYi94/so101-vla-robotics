import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from so101_policies.constants import (
    JOINT_NAMES,
    JOINT_STATE_TOPIC,
    JOINT_TEST_AMPLITUDES,
    LOWER_POSITION_LIMITS,
    POLICY_ACTION_TOPIC,
    UPPER_POSITION_LIMITS,
)


class JointTestPolicy(Node):
    def __init__(self):
        super().__init__("so101_joint_test_policy")

        self.declare_parameter("joint_duration_seconds", 8.0)
        self.declare_parameter("motion_frequency_hz", 0.25)
        self.declare_parameter("publish_rate_hz", 20.0)

        self.joint_duration = self.get_parameter(
            "joint_duration_seconds"
        ).value
        self.motion_frequency = self.get_parameter(
            "motion_frequency_hz"
        ).value
        self.publish_rate = self.get_parameter("publish_rate_hz").value

        if self.joint_duration <= 0.0:
            raise ValueError("joint_duration_seconds must be positive")

        if self.motion_frequency <= 0.0:
            raise ValueError("motion_frequency_hz must be positive")

        if self.publish_rate <= 0.0:
            raise ValueError("publish_rate_hz must be positive")

        self.initial_positions = None
        self.current_joint_index = 0
        self.joint_start_time = None
        self.finished = False

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

    def joint_state_callback(self, message):
        if self.initial_positions is not None:
            return

        if len(message.name) != len(message.position):
            return

        received_positions = dict(zip(message.name, message.position))

        missing_joints = [
            name for name in JOINT_NAMES if name not in received_positions
        ]

        if missing_joints:
            self.get_logger().warning(
                f"Missing joints: {', '.join(missing_joints)}"
            )
            return

        self.initial_positions = [
            received_positions[name] for name in JOINT_NAMES
        ]

        self.joint_start_time = time.monotonic()

        self.get_logger().info(
            f"Starting test for {JOINT_NAMES[0]}."
        )

    def control_loop(self):
        if self.initial_positions is None or self.finished:
            return

        now = time.monotonic()
        elapsed = now - self.joint_start_time

        if elapsed >= self.joint_duration:
            self.advance_to_next_joint(now)
            return

        targets = self.calculate_targets(elapsed)
        self.publish_positions(targets)

    def calculate_targets(self, elapsed):
        targets = self.initial_positions.copy()
        index = self.current_joint_index

        ramp_up = min(elapsed / 1.0, 1.0)
        ramp_down = min((self.joint_duration - elapsed) / 1.0, 1.0)
        envelope = max(0.0, min(ramp_up, ramp_down))

        phase = 2.0 * math.pi * self.motion_frequency * elapsed
        offset = envelope * JOINT_TEST_AMPLITUDES[index] * math.sin(phase)

        targets[index] += offset
        targets[index] = max(
            LOWER_POSITION_LIMITS[index],
            min(UPPER_POSITION_LIMITS[index], targets[index]),
        )

        return targets

    def advance_to_next_joint(self, now):
        self.publish_positions(self.initial_positions)

        self.current_joint_index += 1
        self.joint_start_time = now

        if self.current_joint_index >= len(JOINT_NAMES):
            self.finished = True
            self.get_logger().info(
                "All joint tests completed. The initial pose was restored."
            )
            return

        joint_name = JOINT_NAMES[self.current_joint_index]
        self.get_logger().info(f"Starting test for {joint_name}.")

    def publish_positions(self, positions):
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = JOINT_NAMES.copy()
        message.position = list(positions)

        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = JointTestPolicy()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Joint test stopped by user.")
    finally:
        if node.initial_positions is not None:
            node.publish_positions(node.initial_positions)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()