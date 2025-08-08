from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date
from typing import Iterable, List, Optional, Tuple

import pandas as pd


DB_DIR = os.path.join(os.path.dirname(__file__))
DB_PATH = os.path.abspath(os.path.join(DB_DIR, "feedback.db"))


def _ensure_dir_exists() -> None:
    os.makedirs(DB_DIR, exist_ok=True)


@contextmanager
def get_conn():
    _ensure_dir_exists()
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                sentiment TEXT CHECK(sentiment IN ('positive','negative','neutral')) NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def insert_review(text: str, sentiment: str, created_at: Optional[datetime] = None) -> int:
    if created_at is None:
        created_at = datetime.utcnow()
    created_at_iso = created_at.isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO reviews (text, sentiment, created_at) VALUES (?, ?, ?)",
            (text, sentiment, created_at_iso),
        )
        return int(cur.lastrowid)


def fetch_reviews_between(start_date: date, end_date: date) -> pd.DataFrame:
    start_iso = datetime.combine(start_date, datetime.min.time()).isoformat()
    end_iso = datetime.combine(end_date, datetime.max.time()).isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id, text, sentiment, created_at FROM reviews WHERE created_at BETWEEN ? AND ?",
            (start_iso, end_iso),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["id", "text", "sentiment", "created_at"])
    df = pd.DataFrame(rows, columns=["id", "text", "sentiment", "created_at"])
    # Robust parse for ISO8601 strings with microseconds (and possibly without timezone)
    df["created_at"] = pd.to_datetime(
        df["created_at"], format="ISO8601", errors="coerce"
    )
    df["date"] = df["created_at"].dt.date
    return df


def get_sentiment_counts_by_day(start_date: date, end_date: date) -> pd.DataFrame:
    df = fetch_reviews_between(start_date, end_date)
    if df.empty:
        # Return an empty frame with expected columns
        return pd.DataFrame(columns=["date", "positive", "neutral", "negative"]).astype(
            {"positive": int, "neutral": int, "negative": int}
        )
    counts = (
        df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)[
            ["positive", "neutral", "negative"]
        ]
        if set(["positive", "neutral", "negative"]).issubset(set(df["sentiment"].unique()))
        else df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
    )
    # Ensure all columns exist
    for col in ["positive", "neutral", "negative"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts.reset_index().sort_values("date")
    counts[["positive", "neutral", "negative"]] = counts[[
        "positive",
        "neutral",
        "negative",
    ]].astype(int)
    return counts[["date", "positive", "neutral", "negative"]]

