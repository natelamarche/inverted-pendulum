from collections.abc import Callable

import numpy as np


class Controller:
    def __init__(self, control_fn: Callable[[np.ndarray], float]):
        self.control_fn = control_fn

    def get_action(self, observation: np.ndarray) -> float:
        return float(self.control_fn(observation))
