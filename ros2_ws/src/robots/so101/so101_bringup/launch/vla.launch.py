from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.substitutions import FindExecutable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    checkpoint_path = LaunchConfiguration(
        "checkpoint_path"
    )

    vla_config = PathJoinSubstitution(
        [
            FindPackageShare("so101_bringup"),
            "config",
            "vla.yaml",
        ]
    )

    declare_checkpoint = DeclareLaunchArgument(
        "checkpoint_path",
        description="Path to the trained VLA checkpoint.",
    )

    policy_process = ExecuteProcess(
        cmd=[
            FindExecutable(
                name="so101-vla-policy"
            ),
            "--ros-args",
            "--params-file",
            vla_config,
            "-p",
            [
                "checkpoint_path:=",
                checkpoint_path,
            ],
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            declare_checkpoint,
            policy_process,
        ]
    )