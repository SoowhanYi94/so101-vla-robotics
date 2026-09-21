from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    kinematics_config = PathJoinSubstitution(
        [
            FindPackageShare("so101_description"),
            "config",
            "kinematics.yaml",
        ]
    )

    ik_config = PathJoinSubstitution(
        [
            FindPackageShare("so101_bringup"),
            "config",
            "ik.yaml",
        ]
    )

    kinematics_node = Node(
        package="robot_kinematics",
        executable="kinematics_node",
        name="robot_kinematics",
        output="screen",
        parameters=[
            ik_config,
            {
                "kinematics_config": kinematics_config,
            },
        ],
    )

    return LaunchDescription(
        [
            kinematics_node,
        ]
    )