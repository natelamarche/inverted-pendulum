import numpy as np
from pendulum.dynamics.model import state_derivative
from pendulum.dynamics.parameters import PendulumParameters

def rk4_step(
        state: np.ndarray,
        torque: float,
        dt: float,
        params: PendulumParameters
    ) -> np.ndarray:
        k1 = state_derivative(state, torque, params)
        
        k2 = state_derivative(
            state + dt * 0.5 * k1,
            torque,
            params
        )
        
        k3 = state_derivative(
            state + dt * 0.5 * k2,
            torque,
            params
        )
        
        k4 = state_derivative(
            state + dt * k3,
            torque,
            params
        )
        
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0