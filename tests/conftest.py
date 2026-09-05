import pytest

from pendulum.dynamics.parameters import PendulumParameters


@pytest.fixture
def pendulum_params() -> PendulumParameters:
    return PendulumParameters(
        arm_length=0.2,
        arm_inertia=0.01,
        arm_damping=0.002,
        pendulum_mass=0.1,
        pendulum_com_length=0.15,
        pendulum_com_inertia=0.001,
        pendulum_damping=0.001,
        motor_torque_limit=0.5,
    )
