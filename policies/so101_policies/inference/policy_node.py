import time
from pathlib import Path

import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from so101_policies.inference.checkpoint_loader import (
    load_checkpoint,
)
from so101_policies.inference.ros_image import (
    ros_image_to_tensor,
)


class PolicyNode(Node):
    JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    def __init__(self) -> None:
        super().__init__("so101_vla_policy")

        checkpoint_path = self.declare_parameter(
            "checkpoint_path",
            "",
        ).value

        if not checkpoint_path:
            raise ValueError(
                "checkpoint_path parameter is required."
            )

        self.instruction = self.declare_parameter(
            "instruction",
            "pick and place the object",
        ).value

        image_topic = self.declare_parameter(
            "image_topic",
            "/so101/camera/rgb",
        ).value

        joint_state_topic = self.declare_parameter(
            "joint_state_topic",
            "/so101/joint_states",
        ).value

        action_topic = self.declare_parameter(
            "action_topic",
            "/so101/policy_action",
        ).value

        action_frequency = float(
            self.declare_parameter(
                "action_frequency_hz",
                30.0,
            ).value
        )

        self.observation_timeout = float(
            self.declare_parameter(
                "observation_timeout_seconds",
                0.5,
            ).value
        )

        self.maximum_action_step = float(
            self.declare_parameter(
                "maximum_action_step",
                0.10,
            ).value
        )

        requested_device = self.declare_parameter(
            "device",
            "auto",
        ).value

        self.device = self._select_device(
            requested_device
        )

        self.policy = load_checkpoint(
            Path(checkpoint_path),
            self.device,
        )

        token_ids, text_mask = (
            self.policy.tokenizer.encode(
                self.instruction,
                self.policy.configuration.maximum_text_length,
            )
        )

        self.token_ids = token_ids.unsqueeze(0).to(
            self.device
        )

        self.text_mask = text_mask.unsqueeze(0).to(
            self.device
        )

        self.latest_image = None
        self.latest_state = None

        self.latest_image_time = 0.0
        self.latest_state_time = 0.0

        self.action_chunk = None
        self.action_index = 0

        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self._handle_image,
            qos_profile_sensor_data,
        )

        self.state_subscription = self.create_subscription(
            JointState,
            joint_state_topic,
            self._handle_joint_state,
            qos_profile_sensor_data,
        )

        self.action_publisher = self.create_publisher(
            Float64MultiArray,
            action_topic,
            10,
        )

        self.control_timer = self.create_timer(
            1.0 / action_frequency,
            self._control_step,
        )

        self.get_logger().info(
            f"Loaded VLA checkpoint from {checkpoint_path} "
            f"on {self.device}."
        )

    @staticmethod
    def _select_device(
        requested_device: str,
    ) -> torch.device:
        if requested_device == "auto":
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but is unavailable."
                )

            return torch.device("cuda")

        if requested_device == "cpu":
            return torch.device("cpu")

        raise ValueError(
            f"Unsupported device: {requested_device}"
        )

    def _handle_image(
        self,
        message: Image,
    ) -> None:
        try:
            self.latest_image = ros_image_to_tensor(
                message,
                self.policy.configuration.image_height,
                self.policy.configuration.image_width,
            )

            self.latest_image_time = time.monotonic()

        except ValueError as error:
            self.get_logger().warning(str(error))

    def _handle_joint_state(
        self,
        message: JointState,
    ) -> None:
        name_to_position = dict(
            zip(
                message.name,
                message.position,
            )
        )

        if not all(
            name in name_to_position
            for name in self.JOINT_NAMES
        ):
            return

        self.latest_state = torch.tensor(
            [
                name_to_position[name]
                for name in self.JOINT_NAMES
            ],
            dtype=torch.float32,
        )

        self.latest_state_time = time.monotonic()

    def _observations_are_ready(self) -> bool:
        if self.latest_image is None:
            return False

        if self.latest_state is None:
            return False

        current_time = time.monotonic()

        image_is_fresh = (
            current_time - self.latest_image_time
            <= self.observation_timeout
        )

        state_is_fresh = (
            current_time - self.latest_state_time
            <= self.observation_timeout
        )

        return image_is_fresh and state_is_fresh

    def _predict_action_chunk(self) -> None:
        image = self.latest_image.unsqueeze(0).to(
            self.device,
            non_blocking=True,
        )

        state = self.latest_state.unsqueeze(0).to(
            self.device,
            non_blocking=True,
        )

        state = self.policy.normalizer.normalize_states(
            state
        )

        with torch.inference_mode():
            normalized_actions = self.policy.model(
                images=image,
                token_ids=self.token_ids,
                text_attention_mask=self.text_mask,
                robot_states=state,
            )

            actions = (
                self.policy.normalizer.denormalize_actions(
                    normalized_actions
                )
            )

        self.action_chunk = actions[0].cpu()
        self.action_index = 0

    def _control_step(self) -> None:
        if not self._observations_are_ready():
            return

        if (
            self.action_chunk is None
            or self.action_index
            >= self.action_chunk.shape[0]
        ):
            self._predict_action_chunk()

        desired_action = self.action_chunk[
            self.action_index
        ]

        lower_bound = (
            self.latest_state
            - self.maximum_action_step
        )

        upper_bound = (
            self.latest_state
            + self.maximum_action_step
        )

        safe_action = torch.maximum(
            torch.minimum(
                desired_action,
                upper_bound,
            ),
            lower_bound,
        )

        message = Float64MultiArray()
        message.data = safe_action.tolist()

        self.action_publisher.publish(message)

        self.action_index += 1


def main() -> None:
    rclpy.init()

    node = PolicyNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()