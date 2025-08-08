from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

import dateparser


def _parse_single_date(text: str, default: Optional[date] = None) -> Optional[date]:
    parsed = dateparser.parse(text)
    if parsed is None:
        return default
    return parsed.date()


def parse_date_range(text: str) -> Tuple[date, date]:
    """Parse a human-friendly date range string into (start_date, end_date).

    Supports inputs like:
    - "last 7 days"
    - "June 1 to June 15"
    - "2024-06-01 - 2024-06-15"
    - "yesterday to today"
    Falls back to [today - 7 days, today].
    """
    text_norm = text.strip().lower()
    today = date.today()

    # last N days
    m = re.search(r"last\s+(\d+)\s+days?", text_norm)
    if m:
        n = int(m.group(1))
        start = today - timedelta(days=n - 1 if n > 0 else 0)
        end = today
        return start, end

    # Split by common range separators
    for sep in [" to ", " - ", " until "]:
        if sep in text_norm:
            left, right = text_norm.split(sep, 1)
            start = _parse_single_date(left, default=today - timedelta(days=7))
            end = _parse_single_date(right, default=today)
            if start and end and start <= end:
                return start, end

    # Single date or keyword
    single = _parse_single_date(text_norm)
    if single:
        return single, single

    # Fallback
    return today - timedelta(days=7), today


def human_range_label(start: date, end: date) -> str:
    if start == end:
        return start.strftime("%b %d, %Y")
    return f"{start.strftime('%b %d, %Y')} — {end.strftime('%b %d, %Y')}"

