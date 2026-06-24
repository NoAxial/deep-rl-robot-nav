"""
run.py -- Unified launcher for the Deep RL Navigation Project.

Usage:
  py run.py [COMMAND] [OPTIONS]

Commands:
  train         Train a SAC agent normally (headless)
  view          Visualize a trained agent
  analyze       Generate Matplotlib analytics dashboard
  compare       Compare SAC vs PPO learning curves

Examples:
  py run.py train --config configs/hard.yaml
  py run.py view --model robot_nav_model.zip
  py run.py analyze --model robot_nav_model.zip --episodes 50
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Unified launcher for Deep RL Autonomous Mobile Robot Navigation",
        usage="py run.py <command> [<args>]"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ---- Train ----
    p_train = subparsers.add_parser("train", help="Train the agent headless")
    p_train.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # ---- View ----
    p_view = subparsers.add_parser("view", help="Watch a trained agent navigate")
    p_view.add_argument("--model", type=str, default="robot_nav_model.zip", help="Path to trained model")
    p_view.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p_view.add_argument("--skip-episodes", type=int, default=0, help="Fast-forward N episodes without rendering")

    # ---- Analyze ----
    p_analyze = subparsers.add_parser("analyze", help="Generate analytics dashboard")
    p_analyze.add_argument("--model", type=str, default="robot_nav_model.zip", help="Path to trained model")
    p_analyze.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p_analyze.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate")

    # ---- Compare ----
    p_compare = subparsers.add_parser("compare", help="Compare SAC and PPO")
    p_compare.add_argument("--timesteps", type=int, default=50000, help="Timesteps to train each algorithm")
    p_compare.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    args = parser.parse_args()

    if args.command == "train":
        import train_agent
        # Hack sys.argv so argparse in the target script works if it has one
        sys.argv = ["train_agent.py"]
        if args.config: sys.argv.extend(["--config", args.config])
        train_agent.main()

    elif args.command == "view":
        import visualize
        sys.argv = ["visualize.py"]
        sys.argv.extend(["--model", args.model])
        if args.config: sys.argv.extend(["--config", args.config])
        if args.skip_episodes > 0: sys.argv.extend(["--skip-episodes", str(args.skip_episodes)])
        visualize.main()

    elif args.command == "analyze":
        import analytics
        analytics.generate_analytics(args.model, args.config, args.episodes)

    elif args.command == "compare":
        import compare_algorithms
        compare_algorithms.compare(args.timesteps, args.config)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
