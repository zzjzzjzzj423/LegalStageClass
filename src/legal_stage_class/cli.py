from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

from data.processed.prepare_stard import prepare_stard
from .experiment import ExperimentRunner, RunOptions


def _query_ids(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("provide at least one comma-separated query id")
    return result


def _query_set(value: str) -> list[str]:
    try:
        config = json.loads(Path(value).read_text(encoding="utf-8"))
        ids = config["query_ids"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise argparse.ArgumentTypeError(f"cannot read query_ids from {value}") from exc
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item.strip() for item in ids):
        raise argparse.ArgumentTypeError(f"{value}: query_ids must be a non-empty list of strings")
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError(f"{value}: duplicate query IDs")
    return ids


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
    queries = run.add_mutually_exclusive_group(required=True)
    queries.add_argument("--query-ids", type=_query_ids, help="exact comma-separated bounded query set")
    queries.add_argument("--query-set", dest="query_ids", type=_query_set, help="JSON file with an ordered query_ids list")
    run.add_argument("--run-id", default=None)
    run.add_argument("--run-root", type=Path, default=Path("artifacts/runs"))
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--rrf-constant", type=int, default=60)
    run.add_argument("--domain-initial-n", type=int, default=4)
    run.add_argument("--choice-max-files", type=int, default=255)
    run.add_argument("--selected-files", type=int, default=2, help="file count for Jev PRE")
    run.add_argument("--post-shortlist-n", type=int, default=30)
    run.add_argument("--dense-backend", choices=("local", "siliconflow"), default="local")
    run.add_argument("--file-domains", type=Path, default=Path("data/processed/file_domains.jsonl"))
    run.add_argument("--navigation-model", default="BAAI/bge-small-zh-v1.5")
    run.add_argument("--cache-root", type=Path, default=Path(".cache/legal-stage-class"))
    acceptance = commands.add_parser("acceptance", help="run baseline, PRE, and POST for exactly three live queries")
    acceptance.add_argument("--dataset", type=Path, required=True)
    acceptance.add_argument("--query-ids", type=_query_ids, required=True)
    acceptance.add_argument("--run-root", type=Path, default=Path("artifacts/runs"))
    acceptance.add_argument("--run-prefix", default="acceptance")
    acceptance.add_argument("--domain-initial-n", type=int, default=4)
    acceptance.add_argument("--choice-max-files", type=int, default=255)
    acceptance.add_argument("--selected-files", type=int, default=2)
    acceptance.add_argument("--post-shortlist-n", type=int, default=30)
    acceptance.add_argument("--dense-backend", choices=("local", "siliconflow"), default="local")
    acceptance.add_argument("--file-domains", type=Path, default=Path("data/processed/file_domains.jsonl"))
    acceptance.add_argument("--navigation-model", default="BAAI/bge-small-zh-v1.5")
    acceptance.add_argument("--cache-root", type=Path, default=Path(".cache/legal-stage-class"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "data" and args.data_command == "prepare":
            destination = prepare_stard(args.raw_root, args.output_root)
            print(json.dumps({"dataset": str(destination), "dataset_hash": destination.name}, ensure_ascii=False))
            return
        load_dotenv(dotenv_path=Path(".env"), encoding="utf-8-sig", override=False)
        if args.command == "acceptance":
            if len(args.query_ids) != 3:
                raise ValueError("acceptance requires exactly three fixed query ids")
            destinations = []
            runner = ExperimentRunner()
            for strategy in ("baseline_hybrid", "jev_pre_route", "jev_post_rerank"):
                destinations.append(runner.run(args.dataset, RunOptions(
                    strategy=strategy, query_ids=args.query_ids, run_id=f"{args.run_prefix}-{strategy}", run_root=args.run_root,
                    domain_initial_n=args.domain_initial_n,
                    choice_max_files=args.choice_max_files, selected_files=args.selected_files,
                    post_shortlist_n=args.post_shortlist_n, dense_backend=args.dense_backend, file_domains_path=args.file_domains,
                    navigation_model=args.navigation_model, cache_root=args.cache_root,
                )))
            print(json.dumps({"runs": [str(path) for path in destinations]}, ensure_ascii=False))
            return
        run_id = args.run_id or f"{args.strategy}-{uuid.uuid4().hex[:12]}"
        destination = ExperimentRunner().run(args.dataset, RunOptions(
            strategy=args.strategy, query_ids=args.query_ids, run_id=run_id, run_root=args.run_root,
            top_k=args.top_k, rrf_constant=args.rrf_constant,
            domain_initial_n=args.domain_initial_n, choice_max_files=args.choice_max_files,
            selected_files=args.selected_files, post_shortlist_n=args.post_shortlist_n,
            dense_backend=args.dense_backend, file_domains_path=args.file_domains, navigation_model=args.navigation_model, cache_root=args.cache_root,
        ))
        print(json.dumps({"run": str(destination), "run_id": run_id}, ensure_ascii=False))
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
