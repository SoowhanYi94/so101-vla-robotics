import threading
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState

from so101_policies.constants import (
    JOINT_NAMES,
    JOINT_STATE_TOPIC,
    LOWER_POSITION_LIMITS,
    POLICY_ACTION_TOPIC,
    UPPER_POSITION_LIMITS,
)


HELP_TEXT = """
Commands:
  joint <name|number>   Select a joint
  +                     Increase selected joint by the current step
  -                     Decrease selected joint by the current step
  move <delta>          Move selected joint by a specified amount
  set <value>           Set selected joint to an absolute position
  step <value>          Change the default movement step
  home                  Return to the captured initial pose
  print                 Print the current commanded pose
  save <name>           Save the current pose under a name
  list                  List joint names and current positions
  help                  Show these commands
  quit                  Stop the policy

Joint numbers:
  1  shoulder_pan
  2  shoulder_lift
  3  elbow_flex
  4  wrist_flex
  5  wrist_roll
  6  gripper
"""


class PoseTuningPolicy(Node):
    def __init__(self):
        super().__init__("so101_pose_tuning_policy")

        self.declare_parameter("publish_rate_hz", 60.0)

        self.declare_parameter("default_step", 0.02)

        self.declare_parameter("output_file", "captured_poses.yaml")

        self.publish_rate = float(self.get_parameter("publish_rate_hz").value)

        self.movement_step = float(self.get_parameter("default_step").value)

        self.output_file = Path(self.get_parameter("output_file").value).expanduser()

        if self.publish_rate <= 0.0:
            raise ValueError("publish_rate_hz must be positive")

        if self.movement_step <= 0.0:
            raise ValueError("default_step must be positive")

        self.initial_positions = None

        self.commanded_positions = None

        self.selected_joint_index = 0

        self.position_lock = threading.Lock()

        self.stop_requested = False

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
            self.publish_command,
        )

        self.input_thread = threading.Thread(
            target=self.command_loop,
            daemon=True,
        )

        self.input_thread.start()

        self.get_logger().info(
            f"Waiting for joint states on {JOINT_STATE_TOPIC}."
        )

    def joint_state_callback(self, message):
        if self.initial_positions is not None:
            return

        if len(message.name) != len(message.position):
            return

        received = dict(zip(message.name, message.position))

        missing_joints = [
            name for name in JOINT_NAMES if name not in received
        ]

        if missing_joints:
            return

        positions = [received[name] for name in JOINT_NAMES]

        with self.position_lock:
            self.initial_positions = positions.copy()

            self.commanded_positions = positions.copy()

        self.get_logger().info("Initial SO-101 pose captured.")

        self.print_current_pose()

        print(HELP_TEXT)

    def command_loop(self):
        while not self.stop_requested:
            try:
                command = input("pose> ").strip()
            except EOFError:
                self.request_stop()
                return

            if not command:
                continue

            try:
                self.process_command(command)
            except (ValueError, IndexError) as error:
                print(f"Invalid command: {error}")

    def process_command(self, command):
        parts = command.split()

        operation = parts[0].lower()

        if operation == "joint":
            self.select_joint(parts)
        elif operation == "+":
            self.move_selected_joint(self.movement_step)
        elif operation == "-":
            self.move_selected_joint(-self.movement_step)
        elif operation == "move":
            self.move_selected_joint(float(parts[1]))
        elif operation == "set":
            self.set_selected_joint(float(parts[1]))
        elif operation == "step":
            self.set_movement_step(parts)
        elif operation == "home":
            self.return_home()
        elif operation == "print":
            self.print_current_pose()
        elif operation == "save":
            self.save_pose(parts)
        elif operation == "list":
            self.print_joint_list()
        elif operation == "help":
            print(HELP_TEXT)
        elif operation in {"quit", "exit", "q"}:
            self.request_stop()
        else:
            print(f"Unknown command: {operation}")
            print("Enter 'help' to show available commands.")

    def select_joint(self, parts):
        if len(parts) != 2:
            raise ValueError("usage: joint <name|number>")

        selection = parts[1]

        if selection.isdigit():
            index = int(selection) - 1

            if index < 0 or index >= len(JOINT_NAMES):
                raise ValueError("joint number must be between 1 and 6")
        else:
            if selection not in JOINT_NAMES:
                raise ValueError(f"unknown joint: {selection}")

            index = JOINT_NAMES.index(selection)

        self.selected_joint_index = index

        print(f"Selected joint: {JOINT_NAMES[index]}")

    def move_selected_joint(self, delta):
        self.require_initial_state()

        index = self.selected_joint_index

        with self.position_lock:
            requested = self.commanded_positions[index] + delta

            self.commanded_positions[index] = self.clamp_joint(
                index,
                requested,
            )

            value = self.commanded_positions[index]

        print(f"{JOINT_NAMES[index]}: {value:.6f} rad")

    def set_selected_joint(self, value):
        self.require_initial_state()

        index = self.selected_joint_index

        with self.position_lock:
            self.commanded_positions[index] = self.clamp_joint(
                index,
                value,
            )

            value = self.commanded_positions[index]

        print(f"{JOINT_NAMES[index]}: {value:.6f} rad")

    def set_movement_step(self, parts):
        if len(parts) != 2:
            raise ValueError("usage: step <positive value>")

        step = float(parts[1])

        if step <= 0.0:
            raise ValueError("step must be positive")

        self.movement_step = step

        print(f"Movement step: {self.movement_step:.6f} rad")

    def return_home(self):
        self.require_initial_state()

        with self.position_lock:
            self.commanded_positions = self.initial_positions.copy()

        print("Returned to the captured initial pose.")

    def clamp_joint(self, index, value):
        return max(
            LOWER_POSITION_LIMITS[index],
            min(UPPER_POSITION_LIMITS[index], value),
        )

    def publish_command(self):
        if self.commanded_positions is None:
            return

        with self.position_lock:
            positions = self.commanded_positions.copy()

        message = JointState()

        message.header.stamp = self.get_clock().now().to_msg()

        message.name = JOINT_NAMES.copy()

        message.position = positions

        self.publisher.publish(message)

    def print_current_pose(self):
        self.require_initial_state()

        with self.position_lock:
            positions = self.commanded_positions.copy()

        print("Current commanded pose:")

        for name, value in zip(JOINT_NAMES, positions):
            print(f"  {name}: {value:.6f}")

    def print_joint_list(self):
        self.require_initial_state()

        with self.position_lock:
            positions = self.commanded_positions.copy()

        for index, (name, value) in enumerate(
            zip(JOINT_NAMES, positions),
            start=1,
        ):
            selected = "*" if index - 1 == self.selected_joint_index else " "

            print(f"{selected} {index}: {name:<15} {value:.6f}")

    def save_pose(self, parts):
        if len(parts) != 2:
            raise ValueError("usage: save <pose_name>")

        self.require_initial_state()

        pose_name = parts[1]

        with self.position_lock:
            positions = self.commanded_positions.copy()

        document = self.load_output_document()

        document.setdefault("absolute_positions", True)

        document.setdefault("poses", {})

        document["poses"][pose_name] = {
            name: float(value)
            for name, value in zip(JOINT_NAMES, positions)
        }

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.output_file.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                document,
                file,
                sort_keys=False,
            )

        print(f"Saved pose '{pose_name}' to {self.output_file}")

    def load_output_document(self):
        if not self.output_file.exists():
            return {}

        with self.output_file.open("r", encoding="utf-8") as file:
            document = yaml.safe_load(file)

        return document or {}

    def require_initial_state(self):
        if self.initial_positions is None:
            raise ValueError("joint state has not been received yet")

    def request_stop(self):
        self.stop_requested = True

        print("Stopping pose-tuning policy.")

        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = PoseTuningPolicy()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_requested = True

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()