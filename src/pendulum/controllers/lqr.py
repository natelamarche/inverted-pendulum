import numpy as np

from scipy.linalg import solve_discrete_are, solve_continuous_are

from .linearization import linearize, discretize
from pendulum.dynamics.parameters import PendulumParameters


def solve_discrete_lqr(
    A_d: np.ndarray, B_d: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> np.ndarray:
    P = solve_discrete_are(A_d, B_d, Q, R)
    K = np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)

    return K


def solve_continuous_lqr(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> np.ndarray:
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)

    return K


class LQRController:
    def __init__(
        self,
        params: PendulumParameters,
        Q: np.ndarray,
        R: np.ndarray,
        dt: float = 0.01,
        state_eq: np.ndarray | None = None,
        torque_eq: float = 0.0,
    ):
        if state_eq is None:
            state_eq = np.array([0.0, np.pi, 0.0, 0.0])

        self.state_eq = state_eq.copy()
        self.torque_eq = torque_eq
        self.dt = dt

        self.A, self.B = linearize(self.state_eq, self.torque_eq, params)
        self.A_d, self.B_d = discretize(self.A, self.B, self.dt)

        self.Q = Q
        self.R = R

        self.K = solve_continuous_lqr(self.A, self.B, self.Q, self.R)
        self.K_d = solve_discrete_lqr(self.A_d, self.B_d, self.Q, self.R)

    def get_continuous_action(self, state: np.ndarray) -> float:
        state_error = state - self.state_eq
        state_error[1] = (state_error[1] + np.pi) % (2 * np.pi) - np.pi
        return (self.torque_eq - (self.K @ state_error)).item()

    def get_discrete_action(self, state: np.ndarray) -> float:
        state_error = state - self.state_eq
        state_error[1] = (state_error[1] + np.pi) % (2 * np.pi) - np.pi
        return (self.torque_eq - (self.K_d @ state_error)).item()
