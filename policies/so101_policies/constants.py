JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

LOWER_POSITION_LIMITS = [
    -1.919862,
    -1.745329,
    -1.690000,
    -1.658063,
    -2.743847,
    -0.174533,
]

UPPER_POSITION_LIMITS = [
    1.919862,
    1.745329,
    1.690000,
    1.658063,
    2.841206,
    1.745329,
]

JOINT_TEST_AMPLITUDES = [
    0.12,  # shoulder_pan
    0.05,  # shoulder_lift
    0.05,  # elbow_flex
    0.08,  # wrist_flex
    0.15,  # wrist_roll
    0.10,  # gripper
]

JOINT_STATE_TOPIC = "/so101/joint_states"
POLICY_ACTION_TOPIC = "/so101/policy_action"
CAMERA_RGB_TOPIC = "/so101/camera/rgb"