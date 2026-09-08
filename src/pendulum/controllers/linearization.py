from pendulum.dynamics.parameters import PendulumParameters
from pendulum.dynamics.model import state_derivative

from scipy.signal import cont2discrete

import numpy as np


def linearize(
    state: np.ndarray, torque: float, params: PendulumParameters
) -> tuple[np.ndarray, np.ndarray]:

    x0 = state
    u0 = torque

    epsilon = 1e-6

    A = np.empty((4, 4))

    for i in range(A.shape[1]):
        e = np.zeros(A.shape[1])
        e[i] = 1

        A[:, i] = (
            state_derivative(x0 + epsilon * e, u0, params)
            - state_derivative(x0 - epsilon * e, u0, params)
        ) / (2 * epsilon)

    B = (
        state_derivative(x0, u0 + epsilon, params)
        - state_derivative(x0, u0 - epsilon, params)
    ) / (2 * epsilon)

    B = B.reshape(4, 1)

    return A, B


def discretize(
    A: np.ndarray, B: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    m = B.shape[1]

    C = np.eye(n)
    D = np.zeros((n, m))

    A_d, B_d, _, _, _ = cont2discrete((A, B, C, D), dt, method="zoh")

    return A_d, B_d


def main():
    x_eq = np.array([0.0, np.pi, 0.0, 0.0])
    u_eq = 0.0
    params = PendulumParameters(
        arm_length=0.2,
        arm_inertia=0.01,
        pendulum_mass=0.1,
        pendulum_com_length=0.15,
        pendulum_com_inertia=0.001,
        motor_torque_limit=0.5,
    )

    A, B = linearize(x_eq, u_eq, params)

    print(A)

    print(B)

    controllability = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])

    rank = np.linalg.matrix_rank(controllability)

    print(rank)

    A_d, B_d = discretize(A, B, 0.1)

    print(A_d)

    print(B_d)


if __name__ == "__main__":
    main()
