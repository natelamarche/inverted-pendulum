import numpy as np

from .model import accelerations
from .parameters import PendulumParameters

ACCELERATION_LIMIT = 100.0  # Arm acceleration, rad/s².
SPEED_LIMIT = 6.0 # Arm speed rad/s

def torque_to_acceleration_command(
    state: np.ndarray,
    torque: float,
    params: PendulumParameters,
    dt: float,
    acceleration_limit: float = ACCELERATION_LIMIT,
) -> tuple[float, float]:
    """Return requested and clipped arm acceleration for one control interval."""
    requested, _ = accelerations(state, torque, params)
    min_acceleration = max(-acceleration_limit, (-6.0 - state[2] / dt))
    max_acceleration = min(acceleration_limit, (6.0 - state[2]) / dt)
    commanded = float(np.clip(requested, min_acceleration, max_acceleration))
    return float(requested), commanded
