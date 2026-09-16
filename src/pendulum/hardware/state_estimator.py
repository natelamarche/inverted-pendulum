import numpy as np


class StateEstimator:
    def __init__(
        self,
        phi_sign: float = 1.0,
        theta_sign: float = 1.0,
        phi_offset: float = 0.0,
        theta_offset: float = 0.0,
        velocity_smoothing: float = 0.8,
    ) -> None:
        if phi_sign not in (-1.0, 1.0):
            raise ValueError("phi_sign must be -1 or 1")

        if theta_sign not in (-1.0, 1.0):
            raise ValueError("theta_sign must be -1 or 1")

        if not 0.0 <= velocity_smoothing < 1.0:
            raise ValueError("velocity_smoothing must be in [0, 1)")

        self.phi_sign = phi_sign
        self.theta_sign = theta_sign

        self.phi_offset = phi_offset
        self.theta_offset = theta_offset

        self.velocity_smoothing = velocity_smoothing

        self._previous_phi: float | None = None
        self._previous_theta: float | None = None
        self._previous_timestamp: float | None = None

        self._phi_dot = 0.0
        self._theta_dot = 0.0

    def update(
        self,
        phi: float,
        theta: float,
        timestamp: float,
    ) -> np.ndarray:
        phi = self.phi_sign * phi + self.phi_offset
        theta = self.theta_sign * theta + self.theta_offset

        if not np.all(np.isfinite([phi, theta, timestamp])):
            raise ValueError("Angles and timestamps must be finite")

        if self._previous_timestamp is None:
            self._store(phi, theta, timestamp)

            return np.array(
                [phi, theta, 0.0, 0.0],
                dtype=np.float64,
            )

        dt = timestamp - self._previous_timestamp

        if dt <= 0.0:
            raise ValueError(f"Non-positive timestep: {dt}")

        phi_delta = phi - self._previous_phi

        theta_delta = np.arctan2(
            np.sin(theta - self._previous_theta),
            np.cos(theta - self._previous_theta),
        )

        raw_phi_dot = phi_delta / dt
        raw_theta_dot = theta_delta / dt

        alpha = self.velocity_smoothing

        self._phi_dot = alpha * self._phi_dot + (1.0 - alpha) * raw_phi_dot

        self._theta_dot = alpha * self._theta_dot + (1.0 - alpha) * raw_theta_dot

        self._store(phi, theta, timestamp)

        return np.array(
            [
                phi,
                theta,
                self._phi_dot,
                self._theta_dot,
            ],
            dtype=np.float64,
        )

    def reset(self) -> None:
        self._previous_phi = None
        self._previous_theta = None
        self._previous_timestamp = None

        self._phi_dot = 0.0
        self._theta_dot = 0.0

    def _store(
        self,
        phi: float,
        theta: float,
        timestamp: float,
    ) -> None:
        self._previous_phi = phi
        self._previous_theta = theta
        self._previous_timestamp = timestamp
