import omni.graph.core as og
import omni.kit.app


ROS2_BRIDGE_EXTENSION = "isaacsim.ros2.bridge"
ACTION_GRAPH_PATH = "/SO101_ROS2_ActionGraph"


def enable_ros2_bridge():
    extension_manager = omni.kit.app.get_app().get_extension_manager()

    if extension_manager.is_extension_enabled(ROS2_BRIDGE_EXTENSION):
        return

    extension_manager.set_extension_enabled_immediate(
        ROS2_BRIDGE_EXTENSION,
        True,
    )

    print("Enabled Isaac Sim ROS 2 bridge.")


def create_ros2_joint_bridge(
    articulation_root_path,
    joint_state_topic,
    joint_command_topic,
):
    enable_ros2_bridge()

    og.Controller.edit(
        {
            "graph_path": ACTION_GRAPH_PATH,
            "evaluator_name": "execution",
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                (
                    "ReadSimulationTime",
                    "isaacsim.core.nodes.IsaacReadSimulationTime",
                ),
                (
                    "PublishJointState",
                    "isaacsim.ros2.bridge.ROS2PublishJointState",
                ),
                (
                    "SubscribeJointState",
                    "isaacsim.ros2.bridge.ROS2SubscribeJointState",
                ),
                (
                    "ArticulationController",
                    "isaacsim.core.nodes.IsaacArticulationController",
                ),
            ],
            og.Controller.Keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "PublishJointState.inputs:execIn",
                ),
                (
                    "OnPlaybackTick.outputs:tick",
                    "SubscribeJointState.inputs:execIn",
                ),
                (
                    "OnPlaybackTick.outputs:tick",
                    "ArticulationController.inputs:execIn",
                ),
                (
                    "ReadSimulationTime.outputs:simulationTime",
                    "PublishJointState.inputs:timeStamp",
                ),
                (
                    "SubscribeJointState.outputs:jointNames",
                    "ArticulationController.inputs:jointNames",
                ),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
                (
                    "SubscribeJointState.outputs:velocityCommand",
                    "ArticulationController.inputs:velocityCommand",
                ),
                (
                    "SubscribeJointState.outputs:effortCommand",
                    "ArticulationController.inputs:effortCommand",
                ),
            ],
            og.Controller.Keys.SET_VALUES: [
                (
                    "ArticulationController.inputs:robotPath",
                    articulation_root_path,
                ),
                (
                    "PublishJointState.inputs:targetPrim",
                    [articulation_root_path],
                ),
                (
                    "PublishJointState.inputs:topicName",
                    joint_state_topic,
                ),
                (
                    "SubscribeJointState.inputs:topicName",
                    joint_command_topic,
                ),
            ],
        },
    )

    print(f"Created ROS 2 graph: {ACTION_GRAPH_PATH}")