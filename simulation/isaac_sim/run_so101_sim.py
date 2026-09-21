from isaacsim import SimulationApp


SIMULATION_CONFIG = {
    "headless": False,
    "renderer": "RayTracedLighting",
    "exts": [
        "isaacsim.ros2.bridge",
        "isaacsim.core.nodes",
    ],
}


def main():
    simulation_app = SimulationApp(SIMULATION_CONFIG)

    # Isaac Sim-dependent modules must be imported after SimulationApp starts.
    from so101_scene import SO101Scene

    scene = None

    try:
        scene = SO101Scene(simulation_app)
        scene.initialize()

        print("SO-101 task simulation is running.")
        print("Joint states:  /so101/joint_states")
        print("Joint commands: /so101/joint_commands")
        print("RGB image:     /so101/camera/rgb")
        print("Camera info:   /so101/camera/camera_info")

        while simulation_app.is_running():
            scene.step(render=True)

    except Exception as error:
        print(f"SO-101 simulation failed: {error}")
        raise

    finally:
        if scene is not None:
            scene.close()

        simulation_app.close()
        print("SO-101 simulation closed.")


if __name__ == "__main__":
    main()