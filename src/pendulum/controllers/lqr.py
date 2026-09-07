import numpy as np

from scipy.linalg import solve_continuous_are

from .linearization import linearize
from pendulum.dynamics.parameters import PendulumParameters


def solve_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)

    return K


class LQRController:
    def __init__(
        self,
        params: PendulumParameters,
        Q: np.ndarray,
        R: np.ndarray,
        state_eq: np.ndarray | None = None,
        torque_eq: float = 0.0,
    ):
        if state_eq is None:
            state_eq = np.array([0.0, np.pi, 0.0, 0.0])

        self.state_eq = state_eq.copy()
        self.torque_eq = torque_eq

        self.A, self.B = linearize(self.state_eq, self.torque_eq, params)
        self.Q = Q
        self.R = R

        self.K = solve_lqr(self.A, self.B, self.Q, self.R)

    def get_action(self, state: np.ndarray) -> float:
        state_error = state - self.state_eq
        state_error[1] = (state_error[1] + np.pi) % (2 * np.pi) - np.pi
        return (self.torque_eq - (self.K @ state_error)).item()
