"""
Episodic Memory Ledger (Hippocampal Store)
Append-only store for recording objective execution failures,
AssertionErrors, and spinal interrupts.
"""

import sqlite3
import json
import time
from typing import List, Dict, Any, Optional
from ..types import CerebralState, ActionProposal


class EpisodicLedger:
    """
    Persistent store for empirical failure experiences.
    Never overwrites past history; records raw facts for offline dreaming.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        # Persistent connection for in-memory or file database
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failure_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                session_id TEXT,
                epoch INTEGER,
                state_payload TEXT,
                action_payload TEXT,
                error_type TEXT,
                traceback TEXT,
                resolved INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def record_failure(
        self,
        state: CerebralState,
        action: ActionProposal,
        error_type: str,
        traceback_str: str,
    ) -> int:
        """Log an execution failure as immutable historical fact."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO failure_ledger 
            (timestamp, session_id, epoch, state_payload, action_payload, error_type, traceback, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            time.time(),
            state.session_id,
            state.epoch,
            state.model_dump_json(),
            action.model_dump_json(),
            error_type,
            traceback_str,
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_unresolved_failures(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve unassimilated failure events for the dreaming phase."""
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM failure_ledger WHERE resolved = 0 ORDER BY id ASC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def mark_resolved(self, failure_id: int) -> None:
        """Mark an entry as successfully assimilated into the weights."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE failure_ledger SET resolved = 1 WHERE id = ?", (failure_id,))
        self.conn.commit()
