import matplotlib.pyplot as plt
import numpy as np


def plot_state_history(times: np.ndarray, states: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1)

    axes[0].plot(times, states[:, 0], label="phi")
    axes[0].plot(times, states[:, 1], label="theta")

    axes[0].set_xlabel("Time (s")
    axes[0].set_ylabel("Angle (rad)")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(times, states[:, 2], label="phi_dot")
    axes[1].plot(times, states[:, 3], label="theta_dot")

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angle/Time (rad/s)")
    axes[1].legend()
    axes[1].grid()

    fig.tight_layout()

    plt.show()
