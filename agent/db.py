"""
SQLite database layer for the reputation agent.

Two tables:
  - reviews: every review ever processed (dedup tracker + analytics source)
  - response_queue: negative reviews awaiting human approval
"""

import json
import sqlite3
from datetime import datetime, timezone

import agent_config
DB_PATH = agent_config.DB_PATH


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id       TEXT PRIMARY KEY,
            reviewer_name   TEXT,
            rating          INTEGER,
            review_text     TEXT,
            review_date     TEXT,
            overall_sentiment TEXT,
            aspects         TEXT,
            urgent          INTEGER DEFAULT 0,
            severity_score  REAL DEFAULT 0,
            processed_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS response_queue (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id       TEXT UNIQUE REFERENCES reviews(review_id),
            draft_response  TEXT,
            status          TEXT DEFAULT 'pending',
            created_at      TEXT,
            reviewed_at     TEXT,
            edited_response TEXT
        );
    """)
    conn.commit()
    conn.close()


def get_seen_ids():
    """Return set of all review_ids already in the database."""
    conn = _connect()
    rows = conn.execute("SELECT review_id FROM reviews").fetchall()
    conn.close()
    return {row["review_id"] for row in rows}


def insert_review(record: dict):
    """Insert a processed review into the reviews table."""
    conn = _connect()
    conn.execute(
        """INSERT OR IGNORE INTO reviews
           (review_id, reviewer_name, rating, review_text, review_date,
            overall_sentiment, aspects, urgent, severity_score, processed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            record["review_id"],
            record["reviewer_name"],
            record["rating"],
            record["review_text"],
            record.get("review_date", ""),
            record["overall_sentiment"],
            json.dumps(record.get("aspect_sentiments", {}), ensure_ascii=False),
            1 if record.get("urgent") else 0,
            record.get("severity_score", 0),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def insert_response(review_id: str, draft: str):
    """Insert a draft response into the queue."""
    conn = _connect()
    conn.execute(
        """INSERT OR IGNORE INTO response_queue
           (review_id, draft_response, status, created_at)
           VALUES (?, ?, 'pending', ?)""",
        (review_id, draft, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_pending_responses():
    """Return all pending response queue items joined with review data."""
    conn = _connect()
    rows = conn.execute("""
        SELECT rq.id, rq.review_id, rq.draft_response, rq.status, rq.created_at,
               r.reviewer_name, r.rating, r.review_text, r.review_date,
               r.overall_sentiment, r.urgent, r.severity_score
        FROM response_queue rq
        JOIN reviews r ON rq.review_id = r.review_id
        WHERE rq.status = 'pending'
        ORDER BY r.rating ASC, rq.created_at ASC
    """).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def approve_response(queue_id: int, edited_text: str):
    """Mark a response as approved with the (possibly edited) final text."""
    conn = _connect()
    conn.execute(
        """UPDATE response_queue
           SET status = 'approved', edited_response = ?, reviewed_at = ?
           WHERE id = ?""",
        (edited_text, datetime.now(timezone.utc).isoformat(), queue_id),
    )
    conn.commit()
    conn.close()


def reject_response(queue_id: int):
    """Mark a response as rejected."""
    conn = _connect()
    conn.execute(
        """UPDATE response_queue
           SET status = 'rejected', reviewed_at = ?
           WHERE id = ?""",
        (datetime.now(timezone.utc).isoformat(), queue_id),
    )
    conn.commit()
    conn.close()


def get_all_reviews():
    """Return all reviews for analytics."""
    conn = _connect()
    rows = conn.execute("SELECT * FROM reviews ORDER BY review_date DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_complaint_counts():
    """
    Count negative aspect mentions across all negative reviews.
    Returns list of (aspect, count) sorted by count descending.
    """
    conn = _connect()
    rows = conn.execute(
        "SELECT aspects FROM reviews WHERE rating <= 3"
    ).fetchall()
    conn.close()

    counter = {}
    for row in rows:
        if not row["aspects"]:
            continue
        try:
            aspects = json.loads(row["aspects"])
        except (json.JSONDecodeError, TypeError):
            continue
        for aspect, data in aspects.items():
            if isinstance(data, dict) and data.get("sentiment") == "negative":
                counter[aspect] = counter.get(aspect, 0) + 1

    return sorted(counter.items(), key=lambda x: -x[1])
