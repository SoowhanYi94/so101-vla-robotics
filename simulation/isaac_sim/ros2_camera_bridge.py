import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata
import omni.syntheticdata._syntheticdata as sd


try:
    from isaacsim.ros2.bridge import read_camera_info
except ModuleNotFoundError:
    try:
        from omni.isaac.ros2_bridge import read_camera_info
    except ModuleNotFoundError:
        read_camera_info = None


def set_publish_rate(render_product, render_variable, step_size):
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        render_variable + "IsaacSimulationGate",
        render_product,
    )

    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def create_rgb_publisher(
    camera,
    topic_name,
    frequency,
    rendering_frequency,
):
    render_product = camera._render_product_path
    frame_id = camera.prim_path.split("/")[-1]
    step_size = max(1, round(rendering_frequency / frequency))

    render_variable = (
        omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
            sd.SensorType.Rgb.name
        )
    )

    writer = rep.writers.get(render_variable + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace="",
        queueSize=1,
        topicName=topic_name,
    )
    writer.attach([render_product])

    set_publish_rate(render_product, render_variable, step_size)

    return writer


def create_camera_info_publisher(
    camera,
    topic_name,
    frequency,
    rendering_frequency,
):
    if read_camera_info is None:
        print(
            "Camera-info publishing is unavailable in this Isaac Sim "
            "installation. RGB publishing will continue."
        )
        return None

    render_product = camera._render_product_path
    frame_id = camera.prim_path.split("/")[-1]
    step_size = max(1, round(rendering_frequency / frequency))

    camera_info, _ = read_camera_info(
        render_product_path=render_product
    )

    writer = rep.writers.get("ROS2PublishCameraInfo")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace="",
        queueSize=1,
        topicName=topic_name,
        width=camera_info.width,
        height=camera_info.height,
        projectionType=camera_info.distortion_model,
        k=camera_info.k.reshape([1, 9]),
        r=camera_info.r.reshape([1, 9]),
        p=camera_info.p.reshape([1, 12]),
        physicalDistortionModel=camera_info.distortion_model,
        physicalDistortionCoefficients=camera_info.d,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatchIsaacSimulationGate",
        render_product,
    )

    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return writer


def create_ros2_camera_publishers(
    camera,
    rgb_topic,
    camera_info_topic,
    frequency,
    rendering_frequency,
):
    writers = []

    rgb_writer = create_rgb_publisher(
        camera,
        rgb_topic,
        frequency,
        rendering_frequency,
    )
    writers.append(rgb_writer)

    info_writer = create_camera_info_publisher(
        camera,
        camera_info_topic,
        frequency,
        rendering_frequency,
    )

    if info_writer is not None:
        writers.append(info_writer)
        print(f"Camera info topic: {camera_info_topic}")

    print(f"RGB camera topic: {rgb_topic}")

    return writers