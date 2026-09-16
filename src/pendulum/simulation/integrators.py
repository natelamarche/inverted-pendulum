import numpy as np

from pendulum.dynamics.model import state_derivative, state_derivative_acceleration
from pendulum.dynamics.parameters import PendulumParameters


def rk4_step(
    state: np.ndarray, torque: float, dt: float, params: PendulumParameters
) -> np.ndarray:
    k1 = state_derivative(state, torque, params)

    k2 = state_derivative(state + dt * 0.5 * k1, torque, params)

    k3 = state_derivative(state + dt * 0.5 * k2, torque, params)

    k4 = state_derivative(state + dt * k3, torque, params)

    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def rk4_step_acceleration(
    state: np.ndarray, arm_acceleration: float, dt: float, params: PendulumParameters
) -> np.ndarray:
    """Hold the arm acceleration command constant across all RK4 stages."""
    
    k1 = state_derivative_acceleration(state, arm_acceleration, params)
    
    k2 = state_derivative_acceleration(state + dt * 0.5 * k1, arm_acceleration, params)
    
    k3 = state_derivative_acceleration(state + dt * 0.5 * k2, arm_acceleration, params)
    
    k4 = state_derivative_acceleration(state + dt * k3, arm_acceleration, params)
    
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
