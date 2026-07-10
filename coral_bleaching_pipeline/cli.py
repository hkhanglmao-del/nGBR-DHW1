from __future__ import annotations

import argparse

from .config import apply_fast_dev_run, load_config, select_models
from .train import run_all, run_eda, run_prepare, run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coral bleaching forecasting and monitoring pipeline",
    )
    parser.add_argument(
        "command",
        choices=["prepare", "eda", "train", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model list, e.g. random_forest,xgboost,lstm,st_gnn.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a tiny smoke test with fewer epochs/trees.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    cfg = select_models(cfg, args.models)
    if args.fast_dev_run:
        cfg = apply_fast_dev_run(cfg)

    if args.command == "prepare":
        prepared = run_prepare(cfg)
        print(
            f"Prepared {len(prepared.node_frame):,} node rows, "
            f"{len(prepared.tabular_frame):,} supervised tabular rows."
        )
    elif args.command == "eda":
        run_eda(cfg)
        print(f"EDA figures written to {cfg['eda']['output_dir']}")
    elif args.command == "train":
        leaderboard = run_train(cfg)
        print(leaderboard.to_string(index=False))
    elif args.command == "all":
        leaderboard = run_all(cfg)
        print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()

