"""SQLite storage for the online training platform.

Tables: users, sessions, runs, config. Plain stdlib (sqlite3 + hashlib) so there
are no extra dependencies. WAL mode is enabled for safe concurrent access from
the FastAPI request handlers and the background job workers.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import time
from typing import Dict, List, Optional

from bvr.training_config import PLATFORM_DEFAULTS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DB_PATH = os.path.join(DATA_DIR, "platform.db")

DEFAULT_CONFIG = {
    "class_access_code": "BVR2026",
    "runs_per_window": "10",
    "steps_per_run": "200000",
    "steps_editable": "0",
    "window_hours": "1",
    "max_concurrent": "2",
    "registration_open": "1",
    "rewards_start_zero": "0",
    "reward_defaults_json": "",
    "reward_ranges_json": "",
    "eval_every_rollouts": "8",
    "eval_episodes_per_enemy": "1",
    "live_eval_max_enemies": "4",
    "locked_eval_episodes_per_enemy": "30",
    "analysis_episodes_per_enemy": "10",
    **PLATFORM_DEFAULTS,
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = _connect()
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                pw_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                status TEXT NOT NULL,            -- queued|running|done|error|stopped
                steps INTEGER NOT NULL,
                rewards_json TEXT,
                enemies TEXT,
                model_path TEXT,
                score REAL,
                mission_rate REAL,
                kill_rate REAL,
                survival_rate REAL,
                mean_reward REAL,
                missile_efficiency REAL,
                submitted INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_reports (
                user_id INTEGER PRIMARY KEY,
                data_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )
        for k, v in DEFAULT_CONFIG.items():
            conn.execute("INSERT OR IGNORE INTO config(key, value) VALUES (?, ?)", (k, v))
        try:
            conn.execute("ALTER TABLE users ADD COLUMN quota_reset_at REAL")
        except sqlite3.OperationalError:
            pass
        # Migrations for columns added after the initial schema.
        for col, decl in (
            ("curve_json", "TEXT"),
            ("analysis_json", "TEXT"),
            ("analysis_status", "TEXT"),
            ("run_uid", "TEXT"),
            ("enemy_weights_json", "TEXT"),
        ):
            try:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.execute(
            "UPDATE runs SET run_uid = 'R' || printf('%06d', id) WHERE run_uid IS NULL OR run_uid = ''"
        )
    conn.close()


def _run_uid(run_id: int) -> str:
    return f"R{int(run_id):06d}"


# --- config --------------------------------------------------------------
def get_config() -> Dict[str, str]:
    conn = _connect()
    rows = conn.execute("SELECT key, value FROM config").fetchall()
    conn.close()
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({r["key"]: r["value"] for r in rows})
    return cfg


def set_config(updates: Dict[str, str]) -> None:
    conn = _connect()
    with conn:
        for k, v in updates.items():
            conn.execute(
                "INSERT INTO config(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (k, str(v)),
            )
    conn.close()


# --- auth ----------------------------------------------------------------
def _hash_pw(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", str(password).encode(), str(salt).encode(), 120_000).hex()


def create_user(name: str, password: str, is_admin: bool = False) -> Optional[Dict]:
    name = (name or "").strip()
    password = password or ""
    if not name or not password:
        return None
    salt = secrets.token_hex(16)
    pw_hash = _hash_pw(password, salt)
    conn = _connect()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO users(name, pw_hash, salt, is_admin, created_at) VALUES (?,?,?,?,?)",
                (name, pw_hash, salt, 1 if is_admin else 0, time.time()),
            )
        return get_user(cur.lastrowid)
    except sqlite3.IntegrityError:
        return None  # name already taken
    finally:
        conn.close()


def verify_user(name: str, password: str) -> Optional[Dict]:
    name = (name or "").strip()
    password = password or ""
    if not name or not password:
        return None
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE name=?", (name,)).fetchone()
    conn.close()
    if row is None:
        return None
    if _hash_pw(password, row["salt"]) != row["pw_hash"]:
        return None
    return dict(row)


def get_user(user_id: int) -> Optional[Dict]:
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def user_count() -> int:
    conn = _connect()
    n = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
    conn.close()
    return n


# --- sessions ------------------------------------------------------------
def create_session(user_id: int, days: int = 7) -> str:
    token = secrets.token_urlsafe(32)
    conn = _connect()
    with conn:
        conn.execute("INSERT INTO sessions(token, user_id, expires) VALUES (?,?,?)",
                     (token, user_id, time.time() + days * 86400))
    conn.close()
    return token


def session_user(token: Optional[str]) -> Optional[Dict]:
    if not token:
        return None
    conn = _connect()
    row = conn.execute("SELECT user_id, expires FROM sessions WHERE token=?", (token,)).fetchone()
    conn.close()
    if row is None or row["expires"] < time.time():
        return None
    return get_user(row["user_id"])


def delete_session(token: Optional[str]) -> None:
    if not token:
        return
    conn = _connect()
    with conn:
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.close()


def set_password(name: str, password: str) -> bool:
    """Reset a user's password (admin recovery). Returns False if user missing."""
    name = (name or "").strip()
    password = password or ""
    if not name or not password:
        return False
    salt = secrets.token_hex(16)
    pw_hash = _hash_pw(password, salt)
    conn = _connect()
    with conn:
        cur = conn.execute(
            "UPDATE users SET pw_hash=?, salt=? WHERE name=?",
            (pw_hash, salt, name),
        )
    conn.close()
    return cur.rowcount > 0


# --- runs ----------------------------------------------------------------
def create_run(user_id: int, steps: int, rewards_json: str, enemies: str,
               enemy_weights_json: Optional[str] = None) -> int:
    conn = _connect()
    with conn:
        cur = conn.execute(
            "INSERT INTO runs(user_id, status, steps, rewards_json, enemies, "
            "enemy_weights_json, created_at) VALUES (?,?,?,?,?,?,?)",
            (user_id, "queued", steps, rewards_json, enemies, enemy_weights_json, time.time()),
        )
        run_id = cur.lastrowid
        conn.execute("UPDATE runs SET run_uid=? WHERE id=?", (_run_uid(run_id), run_id))
    conn.close()
    return run_id


def update_run(run_id: int, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    conn = _connect()
    with conn:
        conn.execute(f"UPDATE runs SET {cols} WHERE id=?", (*fields.values(), run_id))
    conn.close()


def get_run(run_id: int) -> Optional[Dict]:
    conn = _connect()
    row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_user_runs(user_id: int, limit: int = 50) -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        """
        SELECT r.*, u.name AS user_name
        FROM runs r JOIN users u ON u.id = r.user_id
        WHERE r.user_id=? ORDER BY r.created_at DESC LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_run(run_id: int) -> bool:
    conn = _connect()
    with conn:
        cur = conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
    conn.close()
    return cur.rowcount > 0


def count_runs_in_window(user_id: int, window_hours: float) -> int:
    """How many runs the user has STARTED within the rolling window (counts
    queued/running/done/error/stopped - i.e. every attempt consumes quota).

    An admin quota reset sets ``quota_reset_at``; runs created before that
    moment no longer count against the window."""
    since = time.time() - window_hours * 3600
    conn = _connect()
    reset_row = conn.execute("SELECT quota_reset_at FROM users WHERE id=?", (user_id,)).fetchone()
    reset_at = (reset_row["quota_reset_at"] if reset_row else None) or 0.0
    since = max(since, float(reset_at))
    n = conn.execute(
        "SELECT COUNT(*) c FROM runs WHERE user_id=? AND created_at>=?",
        (user_id, since),
    ).fetchone()["c"]
    conn.close()
    return n


def reset_user_quota(user_id: int) -> None:
    """Clear the rolling-window run count for a user (admin action)."""
    conn = _connect()
    with conn:
        conn.execute("UPDATE users SET quota_reset_at=? WHERE id=?", (time.time(), user_id))
    conn.close()


def list_users() -> List[Dict]:
    """All users with run tallies and submission, for the admin users panel."""
    conn = _connect()
    rows = conn.execute(
        """
        SELECT u.id, u.name, u.is_admin, u.created_at, u.quota_reset_at,
               COUNT(r.id) AS total_runs,
               SUM(CASE WHEN r.status='done' THEN 1 ELSE 0 END) AS done_runs,
               MAX(CASE WHEN r.submitted=1 THEN r.score END) AS submitted_score,
               (SELECT COUNT(1) FROM user_reports ur WHERE ur.user_id=u.id) AS has_report
        FROM users u
        LEFT JOIN runs r ON r.user_id = u.id
        GROUP BY u.id
        ORDER BY u.is_admin ASC, u.name ASC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_report(user_id: int) -> Optional[Dict]:
    conn = _connect()
    row = conn.execute(
        "SELECT data_json, updated_at FROM user_reports WHERE user_id=?", (user_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {"data": row["data_json"], "updated_at": row["updated_at"]}


def save_user_report(user_id: int, data_json: str) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO user_reports(user_id, data_json, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET data_json=excluded.data_json, updated_at=excluded.updated_at",
            (user_id, data_json, time.time()),
        )
    conn.close()


def list_queued_runs() -> List[Dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM runs WHERE status='queued' ORDER BY created_at ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def user_submitted_run(user_id: int) -> Optional[Dict]:
    """The user's currently submitted run (if any)."""
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM runs WHERE user_id=? AND submitted=1 AND status='done' "
        "ORDER BY score DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def set_submitted_exclusive(user_id: int, run_id: int) -> None:
    """Mark one run as the user's submission (only one active submission each)."""
    conn = _connect()
    with conn:
        conn.execute("UPDATE runs SET submitted=0 WHERE user_id=?", (user_id,))
        conn.execute("UPDATE runs SET submitted=1 WHERE id=? AND user_id=?", (run_id, user_id))
    conn.close()


def leaderboard() -> List[Dict]:
    """Best submitted run per user, ranked by score."""
    conn = _connect()
    rows = conn.execute(
        """
        SELECT u.name AS name, r.score, r.mission_rate, r.kill_rate, r.survival_rate,
               r.mean_reward, r.missile_efficiency, r.finished_at
        FROM runs r JOIN users u ON u.id = r.user_id
        WHERE r.submitted=1 AND r.status='done' AND r.score IS NOT NULL
        ORDER BY r.score DESC, r.mission_rate DESC, r.mean_reward DESC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def submitted_entries() -> List[Dict]:
    """Each user's submitted run with model path + reward weights, for the final
    competition."""
    conn = _connect()
    rows = conn.execute(
        """
        SELECT u.name AS name, r.model_path, r.rewards_json, r.score,
               r.mission_rate, r.kill_rate
        FROM runs r JOIN users u ON u.id = r.user_id
        WHERE r.submitted=1 AND r.status='done' AND r.score IS NOT NULL AND r.model_path IS NOT NULL
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def all_runs(limit: int = 200) -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        """
        SELECT r.*, u.name AS user_name
        FROM runs r JOIN users u ON u.id=r.user_id
        ORDER BY r.created_at DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
