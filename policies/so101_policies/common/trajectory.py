import numpy as np


def minimum_jerk_blend(progress):
    """Return a smooth blend with zero endpoint velocity and acceleration."""
    progress = np.clip(progress, 0.0, 1.0)

    return (
        10.0 * progress**3
        - 15.0 * progress**4
        + 6.0 * progress**5
    )


def interpolate_joint_positions(start, target, progress):
    """Interpolate between two joint vectors using minimum jerk."""
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if start.shape != target.shape:
        raise ValueError("Start and target positions must have equal shapes")

    blend = minimum_jerk_blend(progress)
    return start + blend * (target - start)