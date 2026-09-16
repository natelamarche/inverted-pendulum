import numpy as np

from .model import accelerations
from .parameters import PendulumParameters

ACCELERATION_LIMIT = 100.0  # Arm acceleration, rad/s².
SPEED_LIMIT = 5.0  # Arm speed rad/s


def torque_to_acceleration_command(
    state: np.ndarray,
    torque: float,
    params: PendulumParameters,
    dt: float,
    acceleration_limit: float = ACCELERATION_LIMIT,
) -> tuple[float, float]:
    """Return requested and clipped arm acceleration for one control interval."""
    requested, _ = accelerations(state, torque, params)
    min_acceleration = float(
        np.clip(
            (-SPEED_LIMIT - state[2]) / dt,
            -acceleration_limit,
            acceleration_limit,
        )
    )
    max_acceleration = float(
        np.clip((SPEED_LIMIT - state[2]) / dt, -acceleration_limit, acceleration_limit)
    )
    commanded = float(np.clip(requested, min_acceleration, max_acceleration))
    return float(requested), commanded
