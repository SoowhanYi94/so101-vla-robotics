import numpy as np
import torch
from sensor_msgs.msg import Image
from torch import Tensor
from torch.nn import functional


def ros_image_to_tensor(
    message: Image,
    output_height: int,
    output_width: int,
) -> Tensor:
    channel_counts = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }

    if message.encoding not in channel_counts:
        raise ValueError(
            f"Unsupported image encoding: {message.encoding}"
        )

    channels = channel_counts[message.encoding]

    rows = np.frombuffer(
        message.data,
        dtype=np.uint8,
    ).reshape(
        message.height,
        message.step,
    )

    image = rows[
        :,
        : message.width * channels,
    ].reshape(
        message.height,
        message.width,
        channels,
    )

    if message.encoding in ("bgr8", "bgra8"):
        image = image[:, :, [2, 1, 0]]

    elif message.encoding == "rgba8":
        image = image[:, :, :3]

    elif message.encoding == "mono8":
        image = np.repeat(
            image,
            repeats=3,
            axis=2,
        )

    image = np.ascontiguousarray(image)

    tensor = torch.from_numpy(image).permute(
        2,
        0,
        1,
    ).float() / 255.0

    tensor = functional.interpolate(
        tensor.unsqueeze(0),
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return tensor.clamp(0.0, 1.0)