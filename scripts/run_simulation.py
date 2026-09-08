from pendulum.dynamics.parameters import PendulumParameters
from pendulum.simulation.simulator import Simulator
from pendulum.simulation.visualtization import plot_state_history
from pendulum.controllers.lqr import LQRController

import numpy as np


def main():
    params = PendulumParameters(
        arm_length=0.2,
        arm_inertia=0.01,
        arm_damping=0.002,
        pendulum_mass=0.1,
        pendulum_com_length=0.15,
        pendulum_com_inertia=0.001,
        pendulum_damping=0.001,
        motor_torque_limit=0.5,
    )

    dt = 0.01
    sim = Simulator(params=params, dt=dt)

    initial = np.array([0.0, np.pi * (160 / 180), 0.0, 0.0])

    sim.reset(initial)

    Q = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 20.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1],
        ]
    )

    R = np.array([[4.0]])

    control_stride = 10
    controller = LQRController(params=params, Q=Q, R=R, dt=dt * control_stride)

    time = 10.0
    timestamps = int(round(time / dt))

    state_history = np.empty((timestamps, 4))
    time_history = np.empty(timestamps)

    for i, _ in enumerate(range(timestamps)):
        if i % control_stride == 0:
            torque = controller.get_discrete_action(sim.get_state())

        state_history[i, :] = sim.step(torque=torque)
        time_history[i] = i * dt

    plot_state_history(time_history, state_history)


if __name__ == "__main__":
    main()
