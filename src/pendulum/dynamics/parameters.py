from dataclasses import dataclass


@dataclass
class PendulumParameters:
    arm_length: float
    arm_inertia: float

    pendulum_mass: float
    pendulum_com_length: float
    pendulum_com_inertia: float

    motor_torque_limit: float

    pendulum_damping: float = 0
    arm_damping: float = 0
    gravity: float = 9.80665
