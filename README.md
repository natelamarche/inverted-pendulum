# inverted-pendulum

A Python project for simulating and controlling a rotary inverted pendulum. It combines nonlinear dynamics, classical LQR control, and a custom PyTorch reinforcement learning pipeline for swing-up and balancing experiments, with an interactive 3D viewer and a serial interface for running learned policies on physical hardware.

## Highlights

- Coupled arm and pendulum dynamics with gravity, damping, and fourth-order Runge–Kutta integration
- Continuous and discrete LQR controllers derived from a linearization around the upright equilibrium
- Custom proximal policy optimization (PPO) with generalized advantage estimation and bounded stochastic actions
- Optional LQR imitation pretraining to give the neural actor a starting point for local balancing
- Gymnasium environment with randomized upright and hanging starts, torque limits, and penalties for abrupt control changes
- Shared torque-to-acceleration conversion and actuator limits in training and hardware control
- Interactive 3D playback with angle and torque traces, single-step controls, and adjustable initial conditions
- Serial hardware control targeting 100 Hz, with encoder state estimation and firmware stop diagnostics

## How the control works

The physical state is `[phi, theta, phi_dot, theta_dot]`: the rotary arm angle, pendulum angle, and their angular velocities. Angles are measured in radians, with `theta = 0` hanging down and `theta = pi` upright. The simulator supports both motor torque inputs and prescribed arm acceleration.

The learned controller uses a `5 → 128 → 128 → 1` actor network. Its inputs are the normalized arm angle, sine and cosine of the pendulum angle, and normalized angular velocities. A separate `6 → 64 → 64 → 1` critic also receives the previous action. PPO trains a Gaussian policy with a tanh transform; the viewer and hardware runner use the deterministic tanh-transformed mean.

The action scales to a requested motor torque, which the dynamics model converts to arm acceleration. Training and hardware control use the same conversion, with a 100 rad/s² acceleration limit and a 5 rad/s arm speed limit. The default training environment uses parameters based on measured components, a 0.01-second control interval, and episodes lasting up to 10 seconds. Rewards penalize state error, torque, and changes in torque, while episode termination limits arm travel and detects falls after the pendulum has reached the upright region.

## Install

Requires Python 3.12 or 3.13. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell. The installation includes NumPy, SciPy, Matplotlib, PyTorch, Gymnasium, and the serial interface dependencies. Stable-Baselines3 provides the environment checker; PPO itself is implemented in this repository.

Run the commands below from the repository root. Plotting and interactive playback require a graphical Matplotlib backend.

## Use

Run the standalone LQR simulation and plot the arm and pendulum state histories:

```sh
python scripts/run_simulation.py
```

This demo starts the pendulum 20 degrees from upright and simulates 10 seconds using its own example physical parameters. It does not require a trained model.

Once you have trained a policy, open the 3D viewer:

```sh
python scripts/view_controller.py --checkpoint models/rl_controller_best.pth --seed 0
```

Click **Start** to begin playback, **Step** to advance one control interval, or **Reset** to restart. Use the sliders to change playback speed, initial arm angle, and initial pendulum tilt; drag the 3D scene to rotate the view. Angle error and requested torque are plotted alongside the animation.

Model checkpoints are generated locally and excluded from version control. A fresh checkout needs the training step below before running the RL viewer or hardware controller.

## Train

Train PPO from scratch:

```sh
python scripts/train_rl.py
```

Alternatively, pretrain the actor on LQR actions near upright and zero-action targets around the hanging position, then use that checkpoint to initialize PPO:

```sh
python scripts/pretrain_lqr.py
python scripts/train_rl.py --checkpoint models/lqr_zeros_pretrained.pth
```

Pretraining reports held-out action error and compares the actor with LQR on local balancing trials. It initializes only the actor; the critic remains untrained. Use `python scripts/pretrain_lqr.py --help` to adjust sample counts, epochs, and initial-state bounds.

Each PPO run collects approximately two million environment steps in rollouts of 8,192 steps. Physical parameters and training settings live in [`scripts/train_rl.py`](scripts/train_rl.py); reward weights and termination rules live in [`pendulum_env.py`](src/pendulum/envs/pendulum_env.py).

Training writes the following files under `models/`:

| File | Purpose |
|---|---|
| `rl_controller_best.pth` | Best policy by mean evaluation return during the current run |
| `rl_controller_latest.pth` | Periodic checkpoint saved every 25 iterations |
| `rl_controller.pth` | Final policy and optimizer state |
| `evaluation.csv` | Evaluation metrics recorded initially, every four iterations, and at completion |
| `lqr_zeros_pretrained.pth` | Optional pretrained actor with a fresh PPO optimizer |

Continue training from a saved policy and optimizer:

```sh
python scripts/train_rl.py --checkpoint models/rl_controller_latest.pth
```

Loading a checkpoint starts a new training budget and evaluation history. Each invocation reuses the same output filenames, so copy any checkpoints and metrics you want to retain before starting another run.

## Hardware

The hardware runner requires a compatible rotary pendulum and firmware implementing the serial protocol in [`interface.py`](src/pendulum/hardware/interface.py). It uses 230400 baud, encoder telemetry, and `ZERO`, `START`, `A`, and `STOP` commands. Firmware and mechanical build files are not included in this repository.

With the pendulum hanging down at startup, replace the example port with your device's serial port:

```sh
python scripts/run_hardware.py \
  --port /dev/ttyACM0 \
  --checkpoint models/rl_controller_best.pth \
  --zero-reference down
```

Use `--zero-reference up` if the pendulum is upright during zeroing. The runner sends `ZERO` and begins control automatically after receiving telemetry. Press **Ctrl+C** to exit; the interface sends `STOP` when closing. Hardware geometry, encoder scaling, and angle conventions must match the configuration in [`train_rl.py`](scripts/train_rl.py), [`interface.py`](src/pendulum/hardware/interface.py), and [`state_estimator.py`](src/pendulum/hardware/state_estimator.py).

## Test and evaluate

Run the automated tests:

```sh
python -m pytest
```

The suite covers dynamics, energy, numerical integration, LQR stabilization, environment behavior, rollout returns, policy evaluation, and the serial interface using simulated hardware responses.

During training, evaluation uses deterministic actions across 20 fixed seeds. The CSV records mean return, episode duration, survival rate, upright-start survival rate, pendulum angle RMSE, requested torque RMS, and the policy's pre-tanh action standard deviation. Survival means reaching the episode time limit without termination; use angle error and the viewer to assess balancing quality alongside that metric.

## Project layout

| Path | Purpose |
|---|---|
| `src/pendulum/dynamics` | Physical parameters, nonlinear equations, energy, and actuation limits |
| `src/pendulum/simulation` | RK4 integration, simulation, state plots, and the 3D viewer |
| `src/pendulum/controllers` | Linearization, LQR, and learned policy inference |
| `src/pendulum/envs` | Gymnasium environment, observations, rewards, and reset conditions |
| `src/pendulum/rl` | Actor–critic networks, PPO, rollout buffer, and evaluation |
| `src/pendulum/hardware` | Serial communication, encoder state estimation, and diagnostics |
| `scripts` | Simulation, pretraining, training, playback, and hardware entry points |
| `tests` | Automated physics, control, learning, and interface checks |
| `models` | Locally generated checkpoints and evaluation logs; excluded from version control |

## License

Released under the [MIT License](LICENSE).
