"""Persistent budget shared by all Jev requests from the project root."""

import sqlite3
from contextlib import closing
from pathlib import Path

NANODOLLARS_PER_INPUT_TOKEN = 42
REQUEST_TOKEN_LIMIT = 65_536  # Conservative bound for the default 64k request context.
BUDGET_TOKENS = 24_000_000_000 // NANODOLLARS_PER_INPUT_TOKEN


def cost_usd(tokens: int) -> float:
    return tokens * NANODOLLARS_PER_INPUT_TOKEN / 1_000_000_000


class JevBudget:
    def __init__(self) -> None:
        self.path = Path("artifacts/jev-budget.sqlite3").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS jev_requests (
                    id INTEGER PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    model TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    reserved_tokens INTEGER NOT NULL,
                    input_tokens INTEGER,
                    request_id TEXT
                )
            """)

    def reserve(self, run_id: str, query_id: str, operation: str, model: str) -> int:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            total, = db.execute(
                "SELECT COALESCE(SUM(COALESCE(input_tokens, reserved_tokens)), 0) FROM jev_requests"
            ).fetchone()
            if total + REQUEST_TOKEN_LIMIT > BUDGET_TOKENS:
                raise RuntimeError(f"Jev budget of USD 24 would be exceeded; accounted USD {cost_usd(total):.6f}")
            row = db.execute(
                "INSERT INTO jev_requests (run_id, query_id, operation, model, reserved_tokens) VALUES (?, ?, ?, ?, ?)",
                (run_id, query_id, operation, model, REQUEST_TOKEN_LIMIT),
            )
            return row.lastrowid

    def settle(self, attempt_id: int, input_tokens: int | None, request_id: str | None, model: str) -> tuple[float, float]:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "UPDATE jev_requests SET input_tokens = ?, request_id = ?, model = ? WHERE id = ?",
                (input_tokens, request_id, model, attempt_id),
            )
            confirmed, accounted = db.execute("""
                SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(COALESCE(input_tokens, reserved_tokens)), 0)
                FROM jev_requests
            """).fetchone()
        return cost_usd(confirmed), cost_usd(accounted)
