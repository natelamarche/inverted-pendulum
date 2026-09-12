from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from pendulum.controllers.controller import Controller
from pendulum.dynamics.parameters import PendulumParameters
from pendulum.envs.pendulum_env import PendulumEnv


def pendulum_geometry(state: np.ndarray, params: PendulumParameters) -> np.ndarray:
    phi, theta = state[:2]
    hinge = params.arm_length * np.array([np.cos(phi), np.sin(phi), 0.0])
    offset = params.pendulum_com_length * np.array(
        [-np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta), -np.cos(theta)]
    )
    return np.array([np.zeros(3), hinge, hinge + offset])


class PendulumViewer:
    def __init__(self, env: PendulumEnv, controller: Controller, seed: int = 0):
        self.env = env
        self.controller = controller
        self.seed = seed
        self.paused = True
        self.done = False
        self.speed = 1.0
        self._budget = 0.0
        self.history = deque(maxlen=1000)
        self.figure = plt.figure(figsize=(12, 7))
        grid = self.figure.add_gridspec(2, 2, width_ratios=[1.6, 1])
        self.scene = self.figure.add_subplot(grid[:, 0], projection="3d")
        self.angle_axes = self.figure.add_subplot(grid[0, 1])
        self.torque_axes = self.figure.add_subplot(grid[1, 1])
        self.figure.subplots_adjust(bottom=0.25, top=0.87, wspace=0.3, hspace=0.5)
        self.figure.suptitle("Rotary pendulum · controller inspection")

        reach = max(env.params.arm_length, env.params.pendulum_com_length) * 1.2
        self.scene.set(
            xlim=(-reach, reach),
            ylim=(-reach, reach),
            zlim=(-reach, reach),
            xlabel="x (m)",
            ylabel="y (m)",
            zlabel="z (m)",
            title="Drag to rotate · rod ends at COM",
        )
        self.scene.set_box_aspect((1, 1, 1))
        self.scene.view_init(elev=22, azim=40)
        circle = np.linspace(0, 2 * np.pi, 100)
        self.scene.plot(
            env.params.arm_length * np.cos(circle),
            env.params.arm_length * np.sin(circle),
            np.zeros_like(circle),
            color="0.75",
            linestyle="--",
        )
        (self.arm,) = self.scene.plot([], [], [], "o-", lw=5, color="tab:blue")
        (self.rod,) = self.scene.plot([], [], [], "o-", lw=4, color="tab:orange")
        (self.angle_line,) = self.angle_axes.plot([], [], color="tab:orange")
        (self.torque_line,) = self.torque_axes.plot([], [], color="tab:blue")
        self.angle_axes.set(ylabel="Upright error (deg)", xlabel="Time (s)")
        torque_extent = 1.1 * env.params.motor_torque_limit
        self.torque_axes.set(
            ylabel="Motor torque (Nm)",
            xlabel="Time (s)",
            ylim=(-torque_extent, torque_extent),
        )
        for axes in (self.angle_axes, self.torque_axes):
            axes.set_xlim(0, env.max_episode_steps * env.dt)
            axes.grid(alpha=0.3)
            axes.axhline(0, color="0.6", lw=0.7)
        self.status = self.figure.text(0.08, 0.13, "", family="monospace")
        self.pause_button = Button(
            self.figure.add_axes((0.08, 0.04, 0.12, 0.05)), "Start"
        )
        self.step_button = Button(
            self.figure.add_axes((0.22, 0.04, 0.12, 0.05)), "Step"
        )
        self.reset_button = Button(
            self.figure.add_axes((0.36, 0.04, 0.12, 0.05)), "Reset"
        )
        self.speed_slider = Slider(
            self.figure.add_axes((0.62, 0.05, 0.27, 0.025)),
            "Speed",
            0.1,
            3.0,
            valinit=1.0,
            valfmt="%1.1fx",
        )
        self.env.reset(seed=self.seed)
        initial = self.env.simulator.get_state()
        arm_angle = np.rad2deg(np.arctan2(np.sin(initial[0]), np.cos(initial[0])))
        error = initial[1] - np.pi
        tilt = np.rad2deg(np.arctan2(np.sin(error), np.cos(error)))
        self.arm_slider = Slider(
            self.figure.add_axes((0.62, 0.15, 0.27, 0.025)),
            "Initial arm (deg)",
            -180,
            180,
            valinit=arm_angle,
            valfmt="%.1f°",
        )
        self.tilt_slider = Slider(
            self.figure.add_axes((0.62, 0.10, 0.27, 0.025)),
            "Initial tilt (deg)",
            -180,
            180,
            valinit=tilt,
            valfmt="%.1f°",
        )
        self.arm_slider.on_changed(self.reset)
        self.tilt_slider.on_changed(self.reset)
        self.pause_button.on_clicked(self._toggle_pause)
        self.step_button.on_clicked(self._single_step)
        self.reset_button.on_clicked(self.reset)
        self.speed_slider.on_changed(self._set_speed)
        self.reset()
        self.animation = FuncAnimation(
            self.figure,
            self._tick,
            init_func=self._draw,
            interval=1000 / 30,
            cache_frame_data=False,
        )

    def reset(self, event=None):
        initial_state = np.array(
            [
                np.deg2rad(self.arm_slider.val),
                np.pi + np.deg2rad(self.tilt_slider.val),
                0.0,
                0.0,
            ]
        )
        self.observation, _ = self.env.reset(
            seed=self.seed, options={"initial_state": initial_state}
        )
        self.paused = True
        self.pause_button.label.set_text("Start")
        self.done = False
        self._budget = 0.0
        self.episode_return = 0.0
        self.outcome = "Running"
        self.history.clear()
        self._record(0.0)
        self._draw()

    def _record(self, torque):
        theta = self.env.simulator.get_state()[1]
        error = np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))
        self.history.append((self.env.steps * self.env.dt, np.rad2deg(error), torque))

    def advance(self):
        """Advance one physics/control step, stopping at the episode boundary."""
        if self.done:
            return
        action = float(self.controller.get_action(self.observation.copy()))
        if not np.isfinite(action):
            raise ValueError("Controller returned a non-finite action")
        self.observation, reward, terminated, truncated, _ = self.env.step(
            np.array([action], dtype=np.float32)
        )
        self.episode_return += reward
        self.done = terminated or truncated
        self.outcome = (
            "Failed: angle cutoff"
            if terminated
            else "Complete: time limit"
            if truncated
            else "Running"
        )
        self._record(action * self.env.params.motor_torque_limit)

    def _toggle_pause(self, event=None):
        self.paused = not self.paused
        self.pause_button.label.set_text("Resume" if self.paused else "Pause")
        self._draw()

    def _single_step(self, event=None):
        self.paused = True
        self.pause_button.label.set_text("Resume")
        self.advance()
        self._draw()

    def _set_speed(self, value):
        self.speed = value

    def _tick(self, frame):
        if not self.paused and not self.done:
            self._budget += self.speed / 30
            while self._budget + 1e-12 >= self.env.dt and not self.done:
                self.advance()
                self._budget -= self.env.dt
        self._draw()

    def _draw(self):
        points = pendulum_geometry(self.env.simulator.get_state(), self.env.params)
        self.arm.set_data_3d(*points[:2].T)
        self.rod.set_data_3d(*points[1:].T)
        history = np.asarray(self.history)
        for axes, line, column in (
            (self.angle_axes, self.angle_line, 1),
            (self.torque_axes, self.torque_line, 2),
        ):
            line.set_data(history[:, 0], history[:, column])
            axes.relim()
            axes.autoscale_view(scalex=False)
        state = self.env.simulator.get_state()
        label = self.outcome if self.done else "Paused" if self.paused else "Running"
        self.status.set_text(
            f"{label} | t={history[-1, 0]:.2f}s | return={self.episode_return:.2f}\n"
            f"phi={state[0]:.3f} rad | error={history[-1, 1]:.2f} deg | "
            f"torque={history[-1, 2]:+.4f} Nm"
        )
        self.figure.canvas.draw_idle()
        return self.arm, self.rod, self.angle_line, self.torque_line, self.status

    def show(self):
        plt.show()
