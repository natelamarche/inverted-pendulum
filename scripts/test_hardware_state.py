import numpy as np

from pendulum.hardware.interface import HardwarePendulum
from pendulum.hardware.state_estimator import StateEstimator

estimator = StateEstimator(
    theta_offset=np.pi,
)

with HardwarePendulum(
    port="/dev/cu.usbmodem103",
    state_estimator=estimator
) as hardware:
    while True:
        state = hardware.read_state()
        print(state)