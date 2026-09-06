from .parameters import PendulumParameters

import numpy as np


def kinetic_energy(state: np.ndarray, params: PendulumParameters) -> float:
    (_phi, theta, phi_dot, theta_dot) = state

    r = params.arm_length
    Jr = params.arm_inertia

    m = params.pendulum_mass
    lp = params.pendulum_com_length
    Jp = params.pendulum_com_inertia

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    A = Jr + m * r**2 + m * lp**2 * sin_theta**2
    B = Jp + m * lp**2
    C = m * r * lp * cos_theta

    T = 0.5 * A * phi_dot**2 + 0.5 * B * theta_dot**2 + C * phi_dot * theta_dot

    return float(T)


def potential_energy(state: np.ndarray, params: PendulumParameters) -> float:
    (_phi, theta, _phi_dot, _theta_dot) = state

    m = params.pendulum_mass
    lp = params.pendulum_com_length

    g = params.gravity

    cos_theta = np.cos(theta)

    V = m * g * lp * (1 - cos_theta)

    return float(V)


def total_energy(state: np.ndarray, params: PendulumParameters) -> float:
    return potential_energy(state, params) + kinetic_energy(state, params)
