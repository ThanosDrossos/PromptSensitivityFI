"""SQLite-backed (prompt, model) cache.

Implements the §7.3 hard rule: never re-call a (prompt, model, sampling-params)
tuple already in the database. The cache is keyed by SHA256 of the canonical
JSON of `LLMRequest`, so any change to messages, temperature, model id,
top_logprobs, etc. invalidates a hit — exactly what we want.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .schemas import LLMRequest, LLMResponse


_DDL = """
CREATE TABLE IF NOT EXISTS llm_cache (
    request_hash TEXT PRIMARY KEY,
    provider     TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    purpose      TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_llm_cache_model ON llm_cache(provider, model_id);
CREATE INDEX IF NOT EXISTS idx_llm_cache_purpose ON llm_cache(purpose);
"""


class LLMCache:
    """Thin SQLite wrapper. Thread-safe via a single lock.

    Sized for ~2M rows (pilot run); SQLite handles this comfortably with WAL
    mode, which we enable on first connect.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)
        self._conn.commit()

    def get(self, request: LLMRequest) -> LLMResponse | None:
        key = request.cache_key()
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json FROM llm_cache WHERE request_hash = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        response = LLMResponse.model_validate_json(row[0])
        response.cached = True
        return response

    def put(self, request: LLMRequest, response: LLMResponse) -> None:
        key = request.cache_key()
        if response.request_hash != key:
            response = response.model_copy(update={"request_hash": key})
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                  (request_hash, provider, model_id, purpose, request_json, response_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    request.provider,
                    request.model_id,
                    request.purpose,
                    json.dumps(request.model_dump(mode="json"), sort_keys=True),
                    response.model_dump_json(),
                ),
            )
            self._conn.commit()

    def size(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "LLMCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
