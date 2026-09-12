import argparse
from pathlib import Path

from pendulum.controllers.rl import RLController
from pendulum.simulation.viewer import PendulumViewer
from train_rl import make_env


def main():
    parser = argparse.ArgumentParser(description="Inspect an RL controller in 3D")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/rl_controller_best.pth")
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Initial-state seed; Reset repeats this state",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(
            f"Checkpoint not found: {args.checkpoint}. Run scripts/train_rl.py."
        )
    controller = RLController.from_checkpoint(args.checkpoint)
    env = make_env()
    try:
        viewer = PendulumViewer(env, controller, seed=args.seed)
        viewer.show()
    finally:
        env.close()


if __name__ == "__main__":
    main()
