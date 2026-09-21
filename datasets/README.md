Datasets

Robot demonstrations are kept outside the ROS workspace.

datasets/
├── raw/         # Downloaded or recorded source datasets
└── processed/   # Converted, filtered, or simulation-matched datasets

The current training dataset is kdaterao/so101_data in LeRobot v3 format, stored at
raw/so101_data. The loader uses PyAV for video decoding. The policy adapter selects the camera,
resizes RGB frames, converts joint units when configured, creates future action chunks, and
applies normalization statistics.

Before deploying a downloaded trajectory, verify camera choice, joint order, degrees versus
radians, signs, zero offsets, gripper mapping, frequency, and episode task labels.

Metadata inspection:

find datasets/raw/so101_data/meta -type f | sort
sed -n '1,120p' datasets/raw/so101_data/meta/tasks.jsonl

Datasets are inputs, not source code. Avoid committing large video and Parquet files.