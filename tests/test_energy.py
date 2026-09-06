from dataclasses import replace

import numpy as np

from pendulum.dynamics.energy import kinetic_energy, potential_energy, total_energy
from pendulum.dynamics.parameters import PendulumParameters
from pendulum.simulation.simulator import Simulator


def test_energy_is_zero_at_stationary_downward_equilibrium(
    pendulum_params: PendulumParameters,
) -> None:
    state = np.zeros(4)

    assert kinetic_energy(state, pendulum_params) == 0.0
    assert potential_energy(state, pendulum_params) == 0.0
    assert total_energy(state, pendulum_params) == 0.0


def test_stationary_upright_pendulum_has_expected_potential_energy(
    pendulum_params: PendulumParameters,
) -> None:
    state = np.array([0.0, np.pi, 0.0, 0.0])
    expected_energy = (
        2.0
        * pendulum_params.pendulum_mass
        * pendulum_params.gravity
        * pendulum_params.pendulum_com_length
    )

    energy = potential_energy(state, pendulum_params)

    np.testing.assert_allclose(energy, expected_energy)
    np.testing.assert_allclose(total_energy(state, pendulum_params), expected_energy)


def test_unforced_undamped_simulation_conserves_energy(
    pendulum_params: PendulumParameters,
) -> None:
    undamped_params = replace(
        pendulum_params,
        arm_damping=0.0,
        pendulum_damping=0.0,
    )
    simulator = Simulator(undamped_params, dt=0.001)
    initial_state = np.array([0.0, 0.7, 0.4, -0.2])
    simulator.reset(initial_state)
    initial_energy = total_energy(initial_state, undamped_params)

    energies = [initial_energy]
    for _ in range(10_000):
        state = simulator.step(torque=0.0)
        energies.append(total_energy(state, undamped_params))

    np.testing.assert_allclose(energies, initial_energy, rtol=1e-8, atol=1e-10)


def test_unforced_damped_simulation_loses_energy(
    pendulum_params: PendulumParameters,
) -> None:
    simulator = Simulator(pendulum_params, dt=0.001)
    initial_state = np.array([0.0, 0.7, 0.4, -0.2])
    simulator.reset(initial_state)
    initial_energy = total_energy(initial_state, pendulum_params)

    energies = []
    for _ in range(5_000):
        state = simulator.step(torque=0.0)
        energies.append(total_energy(state, pendulum_params))

    assert energies[-1] < initial_energy
    assert np.all(np.diff([initial_energy, *energies]) <= 1e-10)
