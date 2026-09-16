import numpy as np

from .parameters import PendulumParameters


def state_derivative(
    state: np.ndarray, torque: float, params: PendulumParameters
) -> np.ndarray:
    # state = [arm_angle, pendulum_angle,
    #          arm_angular_velocity, pendulum_angular_velocity]
    # returns: [arm_angular_velocity, pendulum_angular_velocity,
    #           arm_angular_acceleration, pendulum_angular_acceleration]

    (_phi, theta, phi_dot, theta_dot) = state

    r = params.arm_length
    Jr = params.arm_inertia
    br = params.arm_damping

    m = params.pendulum_mass
    lp = params.pendulum_com_length
    Jp = params.pendulum_com_inertia
    bp = params.pendulum_damping

    g = params.gravity

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Equations are defined using Euler-Lagrange equations on Lagrangian

    A = Jr + m * r**2 + m * lp**2 * sin_theta**2

    B = Jp + m * lp**2

    C = m * r * lp * cos_theta

    h1 = (
        2 * m * lp**2 * sin_theta * cos_theta * theta_dot * phi_dot
        - m * r * lp * sin_theta * theta_dot**2
    )

    h2 = (-m) * lp**2 * sin_theta * cos_theta * phi_dot**2 + m * g * lp * sin_theta

    R1 = torque - h1 - br * phi_dot
    R2 = -h2 - bp * theta_dot

    M = np.array([[A, C], [C, B]])

    b = np.array([R1, R2])

    phi_ddot, theta_ddot = np.linalg.solve(M, b)

    return np.array([phi_dot, theta_dot, phi_ddot, theta_ddot])


def accelerations(
    state: np.ndarray, torque: float, params: PendulumParameters
) -> tuple[float, float]:
    derivative = state_derivative(state, torque, params)

    return (derivative[2], derivative[3])


def state_derivative_acceleration(
    state: np.ndarray, arm_acceleration: float, params: PendulumParameters
) -> np.ndarray:
    _, theta, phi_dot, theta_dot = state
    m = params.pendulum_mass
    lp = params.pendulum_com_length
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    B = params.pendulum_com_inertia + m * lp**2
    C = m * params.arm_length * lp * cos_theta
    h2 = (
        -m * lp**2 * sin_theta * cos_theta * phi_dot**2
        + m * params.gravity * lp * sin_theta
    )
    theta_ddot = (
        -h2 - params.pendulum_damping * theta_dot - C * arm_acceleration
    ) / B
    return np.array([phi_dot, theta_dot, arm_acceleration, theta_ddot])
