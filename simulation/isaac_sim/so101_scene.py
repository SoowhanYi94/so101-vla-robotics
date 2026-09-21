from pathlib import Path

import numpy as np
import omni.usd

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from pxr import UsdGeom, UsdLux, UsdPhysics
from ros2_camera_bridge import create_ros2_camera_publishers
from ros2_joint_bridge import create_ros2_joint_bridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SO101_USD_PATH = (
    PROJECT_ROOT
    / "assets"
    / "robots"
    / "so101"
    / "so101_new_calib_physics.usd"
)

SO101_PRIM_PATH = "/World/SO101"
SURFACE_PRIM_PATH = "/World/Task/Surface"
CUBE_PRIM_PATH = "/World/Task/TargetCube"
CAMERA_PRIM_PATH = "/World/Sensors/OverheadCamera"

JOINT_STATE_TOPIC = "/so101/joint_states"
JOINT_COMMAND_TOPIC = "/so101/joint_commands"
CAMERA_RGB_TOPIC = "/so101/camera/rgb"
CAMERA_INFO_TOPIC = "/so101/camera/camera_info"

FLOOR_POSITION = np.array([0.0, 0.0, -0.80])
FLOOR_SCALE = np.array([3.0, 3.0, 0.05])

TABLE_TOP_POSITION = np.array([0.0, 0.0, -0.025])
TABLE_TOP_SCALE = np.array([0.80, 0.60, 0.05])

TABLE_LEG_SCALE = np.array([0.05, 0.05, 0.75])
TABLE_LEG_Z = -0.40

TABLE_LEG_POSITIONS = [
    np.array([0.35, 0.25, TABLE_LEG_Z]),
    np.array([0.35, -0.25, TABLE_LEG_Z]),
    np.array([-0.35, 0.25, TABLE_LEG_Z]),
    np.array([-0.35, -0.25, TABLE_LEG_Z]),
]

CUBE_INITIAL_POSITION = np.array([-0.25, 0.0, 0.025])
CUBE_INITIAL_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])
CUBE_SCALE = np.array([0.05, 0.05, 0.05])

CAMERA_POSITION = np.array([-0.15, 0.0, 0.65])
CAMERA_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])
CAMERA_RESOLUTION = (640, 480)
CAMERA_FREQUENCY = 30


class SO101Scene:
    def __init__(self, simulation_app):
        self.simulation_app = simulation_app

        self.world = World(
            physics_dt=1.0 / 120.0,
            rendering_dt=1.0 / 60.0,
            stage_units_in_meters=1.0,
        )

        self.floor = None
        self.table_top = None
        self.table_legs = []
        self.cube = None
        self.camera = None
        self.camera_writers = []
        self.articulation_root = None

    def initialize(self):
        self._validate_assets()
        self._create_lighting()
        self._create_floor()
        self._create_table()
        self._create_target_cube()
        self._load_robot()
        self._create_camera()

        self._update_application(20)

        self.articulation_root = self._select_articulation_root(
            SO101_PRIM_PATH
        )

        print(f"SO-101 articulation root: {self.articulation_root}")

        create_ros2_joint_bridge(
            articulation_root_path=self.articulation_root,
            joint_state_topic=JOINT_STATE_TOPIC,
            joint_command_topic=JOINT_COMMAND_TOPIC,
        )

        self._update_application(10)

        self.world.reset()
        self.camera.initialize()

        self.camera_writers = create_ros2_camera_publishers(
            camera=self.camera,
            rgb_topic=CAMERA_RGB_TOPIC,
            camera_info_topic=CAMERA_INFO_TOPIC,
            frequency=CAMERA_FREQUENCY,
            rendering_frequency=60,
        )

        self.reset()
        self._update_application(5)
    def _validate_assets(self):
        if not SO101_USD_PATH.is_file():
            raise FileNotFoundError(
                f"SO-101 USD was not found at {SO101_USD_PATH}"
            )

    def _create_lighting(self):
        stage = omni.usd.get_context().get_stage()
    
        dome_light = UsdLux.DomeLight.Define(
            stage,
            "/World/Lights/DomeLight",
        )
        dome_light.GetIntensityAttr().Set(700.0)
        dome_light.GetColorAttr().Set((0.85, 0.90, 1.00))
    
        key_light = UsdLux.DistantLight.Define(
            stage,
            "/World/Lights/KeyLight",
        )
        key_light.GetIntensityAttr().Set(1500.0)
        key_light.GetAngleAttr().Set(1.0)
        key_light.GetColorAttr().Set((1.00, 0.92, 0.82))
    
        transform = UsdGeom.XformCommonAPI(key_light.GetPrim())
        transform.SetRotate(
            (45.0, 0.0, 35.0),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )


    def _create_floor(self):
        self.floor = self.world.scene.add(
            FixedCuboid(
                prim_path="/World/Environment/Floor",
                name="floor",
                position=FLOOR_POSITION,
                scale=FLOOR_SCALE,
                size=1.0,
                color=np.array([0.18, 0.18, 0.20]),
            )
        )


    def _create_table(self):
        self.table_top = self.world.scene.add(
            FixedCuboid(
                prim_path="/World/Task/TableTop",
                name="table_top",
                position=TABLE_TOP_POSITION,
                scale=TABLE_TOP_SCALE,
                size=1.0,
                color=np.array([0.50, 0.32, 0.18]),
            )
        )

        for index, position in enumerate(TABLE_LEG_POSITIONS):
            leg = self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/Task/TableLeg{index + 1}",
                    name=f"table_leg_{index + 1}",
                    position=position,
                    scale=TABLE_LEG_SCALE,
                    size=1.0,
                    color=np.array([0.30, 0.18, 0.10]),
                )
            )

            self.table_legs.append(leg)
    def _create_target_cube(self):
        self.cube = self.world.scene.add(
            DynamicCuboid(
                prim_path=CUBE_PRIM_PATH,
                name="target_cube",
                position=CUBE_INITIAL_POSITION,
                orientation=CUBE_INITIAL_ORIENTATION,
                scale=CUBE_SCALE,
                size=1.0,
                color=np.array([1.0, 0.05, 0.05]),
                mass=0.05,
            )
        )

        self.cube.set_default_state(
            position=CUBE_INITIAL_POSITION,
            orientation=CUBE_INITIAL_ORIENTATION,
        )

    def _load_robot(self):
        print(f"Loading SO-101 USD: {SO101_USD_PATH}")

        add_reference_to_stage(
            usd_path=str(SO101_USD_PATH),
            prim_path=SO101_PRIM_PATH,
        )

    def _create_camera(self):
        # An identity orientation points the camera along its local -Z axis,
        # producing an overhead view in this placement.
        self.camera = Camera(
            prim_path=CAMERA_PRIM_PATH,
            name="overhead_camera",
            position=CAMERA_POSITION,
            orientation=CAMERA_ORIENTATION,
            frequency=CAMERA_FREQUENCY,
            resolution=CAMERA_RESOLUTION,
        )

    def _find_articulation_roots(self, parent_path):
        stage = omni.usd.get_context().get_stage()
        roots = []

        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())

            belongs_to_robot = (
                prim_path == parent_path
                or prim_path.startswith(parent_path + "/")
            )

            if belongs_to_robot and prim.HasAPI(
                UsdPhysics.ArticulationRootAPI
            ):
                roots.append(prim_path)

        return roots

    def _select_articulation_root(self, parent_path):
        roots = self._find_articulation_roots(parent_path)

        if not roots:
            raise RuntimeError(
                f"No articulation root was found below {parent_path}"
            )

        if len(roots) > 1:
            print("Multiple articulation roots were found:")

            for root in roots:
                print(f"  {root}")

            print(f"Using: {roots[0]}")

        return roots[0]

    def _update_application(self, count):
        for _ in range(count):
            self.simulation_app.update()

    def reset(self):
        """Restore a deterministic initial scene state."""
        self.world.reset()

        self.cube.set_world_pose(
            position=CUBE_INITIAL_POSITION,
            orientation=CUBE_INITIAL_ORIENTATION,
        )

        self.cube.set_linear_velocity(np.zeros(3))
        self.cube.set_angular_velocity(np.zeros(3))

        print("SO-101 task scene reset.")

    def step(self, render=True):
        self.world.step(render=render)

    def get_rgb_image(self):
        rgba = self.camera.get_rgba()

        if rgba is None or rgba.size == 0:
            return None

        return rgba[:, :, :3].copy()

    def get_cube_pose(self):
        position, orientation = self.cube.get_world_pose()

        return {
            "position": position.copy(),
            "orientation": orientation.copy(),
        }

    def get_task_state(self):
        """Return privileged state for scripted experts and evaluation."""
        return {
            "cube_pose": self.get_cube_pose(),
            "success": self.is_task_successful(),
        }

    def is_task_successful(self):
        cube_position, _ = self.cube.get_world_pose()

        # Initial success criterion: the cube was lifted 10 cm.
        return bool(cube_position[2] > 0.10)

    def close(self):
        for writer in self.camera_writers:
            try:
                writer.detach()
            except Exception:
                pass