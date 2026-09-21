from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("so101_control"))
    config_file = package_share / "config" / "controller.yaml"

    controller = Node(
        package="so101_control",
        executable="action_controller",
        name="so101_action_controller",
        output="screen",
        parameters=[str(config_file)],
    )

    return LaunchDescription([controller])