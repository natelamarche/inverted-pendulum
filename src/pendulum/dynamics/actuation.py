import numpy as np

from .model import accelerations
from .parameters import PendulumParameters

ACCELERATION_LIMIT = 100.0  # Arm acceleration, rad/s².


def torque_to_acceleration_command(
    state: np.ndarray,
    torque: float,
    params: PendulumParameters,
    acceleration_limit: float = ACCELERATION_LIMIT,
) -> tuple[float, float]:
    """Return requested and clipped arm acceleration for one control interval."""
    requested, _ = accelerations(state, torque, params)
    commanded = float(np.clip(requested, -acceleration_limit, acceleration_limit))
    return float(requested), commanded
