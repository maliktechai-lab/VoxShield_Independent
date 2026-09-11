"""
VoxShield Incident Logger
==========================

Logs all inference events to a local SQLite database.
No external services. No cloud. All data stays local.

Database: backend/voxshield_incidents.db
  (path configurable via VOXSHIELD_DB environment variable)

Schema:
    incidents table:
        id             INTEGER PK AUTOINCREMENT
        timestamp      TEXT    (ISO 8601 UTC)
        session_id     TEXT    (request session identifier)
        classification TEXT    (SPOOF | BONA_FIDE | ERROR | UNAVAILABLE)
        spoof_prob     REAL
        bf_prob        REAL
        confidence     REAL
        risk_score     REAL
        threat_level   TEXT    (LOW | MEDIUM | HIGH | CRITICAL | UNKNOWN)
        threshold      REAL
        inference_mode TEXT    (asvspoof5 | demo | unavailable)
        latency_ms     REAL
        device         TEXT
        model_version  TEXT
        filename       TEXT    (original upload filename, sanitized)
        file_size_kb   REAL
        error_msg      TEXT    (NULL if no error)
        action_taken   TEXT    (recommended_action from risk engine)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Path resolution
# ──────────────────────────────────────────────────────────────────────────────

def _db_path() -> Path:
    env_val = os.environ.get("VOXSHIELD_DB", "").strip()
    if env_val:
        return Path(env_val)
    # Default: alongside backend/
    return Path(__file__).parent / "voxshield_incidents.db"


# ──────────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS incidents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    session_id      TEXT    NOT NULL,
    classification  TEXT,
    spoof_prob      REAL,
    bf_prob         REAL,
    confidence      REAL,
    risk_score      REAL,
    threat_level    TEXT,
    threshold       REAL,
    inference_mode  TEXT,
    latency_ms      REAL,
    device          TEXT,
    model_version   TEXT,
    filename        TEXT,
    file_size_kb    REAL,
    error_msg       TEXT,
    action_taken    TEXT
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_incidents_ts   ON incidents(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_incidents_cls  ON incidents(classification);",
    "CREATE INDEX IF NOT EXISTS idx_incidents_thr  ON incidents(threat_level);",
]


# ──────────────────────────────────────────────────────────────────────────────
# Logger class
# ──────────────────────────────────────────────────────────────────────────────

class IncidentLogger:
    """Thread-safe SQLite-backed incident logger."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                conn.execute(idx_sql)
            conn.commit()
        logger.info(f"Incident database: {self.db_path}")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_incident(
        self,
        inference_result: Dict,
        filename: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """
        Log a single inference event.

        Args:
            inference_result: Dict returned by InferenceEngine.predict()
            filename:         Original filename (will be sanitized).
            file_size_bytes:  Upload size in bytes.
            session_id:       Optional session identifier.

        Returns:
            Database row id of the inserted incident.
        """
        timestamp  = datetime.now(timezone.utc).isoformat()
        session_id = session_id or str(uuid.uuid4())[:8]

        # Sanitize filename (no path traversal)
        safe_filename = None
        if filename:
            safe_filename = Path(filename).name[:255]  # basename only, max 255 chars

        file_size_kb = (
            round(file_size_bytes / 1024, 2) if file_size_bytes else None
        )

        row = {
            "timestamp":      timestamp,
            "session_id":     session_id,
            "classification": inference_result.get("classification"),
            "spoof_prob":     inference_result.get("spoof_probability"),
            "bf_prob":        inference_result.get("bona_fide_probability"),
            "confidence":     inference_result.get("confidence"),
            "risk_score":     inference_result.get("risk_score"),
            "threat_level":   inference_result.get("threat_level"),
            "threshold":      inference_result.get("decision_threshold"),
            "inference_mode": inference_result.get("inference_mode"),
            "latency_ms":     inference_result.get("latency_ms"),
            "device":         inference_result.get("device"),
            "model_version":  inference_result.get("model_version"),
            "filename":       safe_filename,
            "file_size_kb":   file_size_kb,
            "error_msg":      inference_result.get("error"),
            "action_taken":   inference_result.get("recommended_action"),
        }

        sql = """
        INSERT INTO incidents
            (timestamp, session_id, classification, spoof_prob, bf_prob,
             confidence, risk_score, threat_level, threshold, inference_mode,
             latency_ms, device, model_version, filename, file_size_kb,
             error_msg, action_taken)
        VALUES
            (:timestamp, :session_id, :classification, :spoof_prob, :bf_prob,
             :confidence, :risk_score, :threat_level, :threshold, :inference_mode,
             :latency_ms, :device, :model_version, :filename, :file_size_kb,
             :error_msg, :action_taken)
        """

        with self._conn() as conn:
            cur = conn.execute(sql, row)
            conn.commit()
            return cur.lastrowid

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_incidents(
        self,
        limit: int = 100,
        offset: int = 0,
        classification: Optional[str] = None,
        threat_level: Optional[str] = None,
    ) -> List[Dict]:
        """Return recent incidents as list of dicts."""
        conditions = []
        params: list = []
        if classification:
            conditions.append("classification = ?")
            params.append(classification.upper())
        if threat_level:
            conditions.append("threat_level = ?")
            params.append(threat_level.upper())

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
        SELECT * FROM incidents
        {where}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> Dict:
        """Return aggregate statistics over all logged incidents."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
            if total == 0:
                return {
                    "total": 0,
                    "by_classification": {},
                    "by_threat": {},
                    "avg_latency_ms": None,
                    "avg_spoof_prob": None,
                    "spoof_rate": None,
                    "latest_timestamp": None,
                }

            by_cls = {}
            for row in conn.execute(
                "SELECT classification, COUNT(*) FROM incidents GROUP BY classification"
            ):
                by_cls[row[0]] = row[1]

            by_thr = {}
            for row in conn.execute(
                "SELECT threat_level, COUNT(*) FROM incidents GROUP BY threat_level"
            ):
                by_thr[row[0]] = row[1]

            agg = conn.execute(
                "SELECT AVG(latency_ms), AVG(spoof_prob) FROM incidents "
                "WHERE classification NOT IN ('ERROR','UNAVAILABLE')"
            ).fetchone()

            latest = conn.execute(
                "SELECT timestamp FROM incidents ORDER BY id DESC LIMIT 1"
            ).fetchone()

            n_spoof = by_cls.get("SPOOF", 0)
            n_valid = total - by_cls.get("ERROR", 0) - by_cls.get("UNAVAILABLE", 0)
            spoof_rate = round(n_spoof / n_valid, 4) if n_valid > 0 else None

        return {
            "total":               total,
            "by_classification":   by_cls,
            "by_threat":           by_thr,
            "avg_latency_ms":      round(agg[0], 2) if agg[0] else None,
            "avg_spoof_prob":      round(agg[1], 4) if agg[1] else None,
            "spoof_rate":          spoof_rate,
            "latest_timestamp":    latest[0] if latest else None,
        }

    def get_recent_timeline(self, n: int = 20) -> List[Dict]:
        """Return last n incidents for the live timeline display."""
        sql = """
        SELECT id, timestamp, classification, spoof_prob, risk_score,
               threat_level, latency_ms, filename, inference_mode
        FROM incidents
        ORDER BY id DESC
        LIMIT ?
        """
        with self._conn() as conn:
            rows = conn.execute(sql, [n]).fetchall()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────────────────────────────────────

_logger_instance: Optional[IncidentLogger] = None


def get_incident_logger() -> IncidentLogger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = IncidentLogger()
    return _logger_instance
