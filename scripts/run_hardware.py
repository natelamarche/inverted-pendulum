import argparse
import logging
import time
from pathlib import Path

import numpy as np

from pendulum.controllers.rl import RLController
from pendulum.dynamics.actuation import torque_to_acceleration_command
from pendulum.hardware.interface import HardwarePendulum
from pendulum.hardware.state_estimator import StateEstimator
from train_rl import make_env


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Run RL hardware control at 100 Hz")
    parser.add_argument("--port", required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/rl_controller_best.pth")
    )
    parser.add_argument(
        "--zero-reference",
        choices=["up", "down"],
        required=True,
        help="Pendulum position during ZERO",
    )
    args = parser.parse_args()

    controller = RLController.from_checkpoint(args.checkpoint)

    env = make_env()
    params, period = env.params, env.dt

    # Q = np.array(
    #         [
    #             [1.0, 0.0, 0.0, 0.0],
    #             [0.0, 20.0, 0.0, 0.0],
    #             [0.0, 0.0, 0.1, 0.0],
    #             [0.0, 0.0, 0.0, 0.1],
    #         ]
    #     )

    # R = np.array([[4.0]])

    # controller = LQRController(params=params, Q=Q, R=R, dt=period)

    env.close()
    estimator = StateEstimator(
        theta_offset=np.pi if args.zero_reference == "down" else 0.0,
    )

    try:
        with HardwarePendulum(args.port, estimator) as hardware:
            hardware.zero()
            previous_action = 0.0
            first_action = True
            next_action = time.monotonic()

            while True:
                state = hardware.read_state(timeout=period)
                phi, theta, phi_dot, theta_dot = state
                # Same observation normalization as PendulumEnv.
                observation = np.array(
                    [
                        phi / (2 * np.pi),
                        np.sin(theta),
                        np.cos(theta),
                        phi_dot / 10.0,
                        theta_dot / 15.0,
                        previous_action,
                    ],
                    dtype=np.float32,
                )
                action = controller.get_action(observation)
                torque = action * params.motor_torque_limit
                # The policy requests torque; the hardware accepts acceleration.
                _, acceleration = torque_to_acceleration_command(
                    state, torque, params, period
                )
                if first_action:
                    hardware.start(acceleration)
                    first_action = False
                else:
                    hardware.apply_acceleration(acceleration)
                previous_action = action

                next_action += period
                now = time.monotonic()
                if next_action <= now:
                    next_action = now + period  # Skip missed slots; no catch-up burst.
                time.sleep(next_action - now)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
