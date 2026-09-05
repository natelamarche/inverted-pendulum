import numpy as np

from pendulum.dynamics.parameters import PendulumParameters

from .integrators import rk4_step


class Simulator:
    def __init__(
        self,
        params: PendulumParameters,
        dt: float,
    ):
        self.params = params
        self.dt = dt
        self.state = np.zeros(4)

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is None:
            state = np.zeros(4)

        self.state = state.copy()
        return self.state.copy()

    def step(self, torque: float) -> np.ndarray:
        self.state = rk4_step(self.state, torque, self.dt, self.params)
        return self.state.copy()

    def get_state(self) -> np.ndarray:
        return self.state.copy()
