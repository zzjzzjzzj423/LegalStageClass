from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from data.processed.prepare_stard import prepare_stard
from .experiment import ExperimentRunner, RunOptions


def _query_ids(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("provide at least one comma-separated query id")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="legal-stage-class")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="normalize STARD and derive law_catalog")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    prepare = data_commands.add_parser("prepare")
    prepare.add_argument("--raw-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, default=Path("data/processed/stard"))
    run = commands.add_parser("run", help="run exactly one independent experiment arm")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--strategy", choices=("baseline_hybrid", "jev_pre_route", "jev_post_rerank"), required=True)
    run.add_argument("--query-ids", type=_query_ids, required=True, help="exact comma-separated bounded query set")
    run.add_argument("--run-id", default=None)
    run.add_argument("--run-root", type=Path, default=Path("artifacts/runs"))
    run.add_argument("--candidate-k", type=int, default=1000)
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--rrf-constant", type=int, default=60)
    run.add_argument("--route-threshold", type=float, default=0.5)
    run.add_argument("--min-laws", type=int, default=1)
    run.add_argument("--max-laws", type=int, default=20)
    run.add_argument("--jev-batch-size", type=int, default=32)
    run.add_argument("--pre-cost-ceiling-usd", type=float, default=20.0)
    run.add_argument("--cache-root", type=Path, default=Path(".cache/legal-stage-class"))
    acceptance = commands.add_parser("acceptance", help="run baseline, PRE, and POST for exactly three live queries")
    acceptance.add_argument("--dataset", type=Path, required=True)
    acceptance.add_argument("--query-ids", type=_query_ids, required=True)
    acceptance.add_argument("--run-root", type=Path, default=Path("artifacts/runs"))
    acceptance.add_argument("--run-prefix", default="acceptance")
    acceptance.add_argument("--candidate-k", type=int, default=1000)
    acceptance.add_argument("--route-threshold", type=float, default=0.5)
    acceptance.add_argument("--min-laws", type=int, default=1)
    acceptance.add_argument("--max-laws", type=int, default=20)
    acceptance.add_argument("--jev-batch-size", type=int, default=32)
    acceptance.add_argument("--pre-cost-ceiling-usd", type=float, default=20.0)
    acceptance.add_argument("--cache-root", type=Path, default=Path(".cache/legal-stage-class"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "data" and args.data_command == "prepare":
            destination = prepare_stard(args.raw_root, args.output_root)
            print(json.dumps({"dataset": str(destination), "dataset_hash": destination.name}, ensure_ascii=False))
            return
        if args.command == "acceptance":
            if len(args.query_ids) != 3:
                raise ValueError("acceptance requires exactly three fixed query ids")
            destinations = []
            runner = ExperimentRunner()
            for strategy in ("baseline_hybrid", "jev_pre_route", "jev_post_rerank"):
                destinations.append(runner.run(args.dataset, RunOptions(
                    strategy=strategy, query_ids=args.query_ids, run_id=f"{args.run_prefix}-{strategy}", run_root=args.run_root,
                    candidate_k=args.candidate_k, route_threshold=args.route_threshold, min_laws=args.min_laws,
                    max_laws=args.max_laws, jev_batch_size=args.jev_batch_size, pre_cost_ceiling_usd=args.pre_cost_ceiling_usd,
                    cache_root=args.cache_root,
                )))
            print(json.dumps({"runs": [str(path) for path in destinations]}, ensure_ascii=False))
            return
        run_id = args.run_id or f"{args.strategy}-{uuid.uuid4().hex[:12]}"
        destination = ExperimentRunner().run(args.dataset, RunOptions(
            strategy=args.strategy, query_ids=args.query_ids, run_id=run_id, run_root=args.run_root,
            candidate_k=args.candidate_k, top_k=args.top_k, rrf_constant=args.rrf_constant,
            route_threshold=args.route_threshold, min_laws=args.min_laws, max_laws=args.max_laws,
            jev_batch_size=args.jev_batch_size, pre_cost_ceiling_usd=args.pre_cost_ceiling_usd,
            cache_root=args.cache_root,
        ))
        print(json.dumps({"run": str(destination), "run_id": run_id}, ensure_ascii=False))
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
