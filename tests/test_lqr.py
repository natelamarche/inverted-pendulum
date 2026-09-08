import numpy as np

from pendulum.controllers.linearization import discretize
from pendulum.controllers.lqr import LQRController
from pendulum.dynamics.parameters import PendulumParameters
from pendulum.simulation.simulator import Simulator


def make_controller(
    params: PendulumParameters,
    dt: float = 0.1,
) -> LQRController:
    Q = np.diag([1.0, 20.0, 0.1, 0.1])
    R = np.array([[4.0]])
    return LQRController(params=params, Q=Q, R=R, dt=dt)


def test_zoh_discretization_of_double_integrator() -> None:
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])

    A_d, B_d = discretize(A, B, dt=0.1)

    np.testing.assert_allclose(A_d, [[1.0, 0.1], [0.0, 1.0]])
    np.testing.assert_allclose(B_d, [[0.005], [0.1]])


def test_discrete_lqr_closed_loop_poles_are_stable(
    pendulum_params: PendulumParameters,
) -> None:
    controller = make_controller(pendulum_params)

    closed_loop = controller.A_d - controller.B_d @ controller.K_d
    poles = np.linalg.eigvals(closed_loop)

    assert np.all(np.abs(poles) < 1.0)


def test_discrete_action_is_zero_at_equilibrium(
    pendulum_params: PendulumParameters,
) -> None:
    controller = make_controller(pendulum_params)

    action = controller.get_discrete_action(controller.state_eq)

    assert action == 0.0


def test_discrete_action_wraps_equivalent_pendulum_angles(
    pendulum_params: PendulumParameters,
) -> None:
    controller = make_controller(pendulum_params)
    state_above_positive_pi = np.array([0.0, np.pi + 0.05, 0.0, 0.0])
    equivalent_state = np.array([0.0, -np.pi + 0.05, 0.0, 0.0])

    action = controller.get_discrete_action(state_above_positive_pi)
    equivalent_action = controller.get_discrete_action(equivalent_state)

    np.testing.assert_allclose(action, equivalent_action)


def test_sampled_controller_stabilizes_near_upright_equilibrium(
    pendulum_params: PendulumParameters,
) -> None:
    simulation_dt = 0.01
    control_stride = 10
    controller = make_controller(
        pendulum_params,
        dt=simulation_dt * control_stride,
    )
    simulator = Simulator(pendulum_params, dt=simulation_dt)
    initial_state = np.array([0.0, np.pi + 0.05, 0.0, 0.0])
    simulator.reset(initial_state)
    torque = 0.0

    for step in range(500):
        if step % control_stride == 0:
            torque = controller.get_discrete_action(simulator.get_state())
        simulator.step(torque)

    final_error = simulator.get_state() - controller.state_eq

    assert np.linalg.norm(final_error) < 1e-5
