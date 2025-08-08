from __future__ import annotations

import random
from datetime import datetime, timedelta

from app.data.db import init_db, insert_review


SAMPLE_POSITIVE = [
    "The ramen was incredible and the service was fast!",
    "Loved the spicy broth. Staff were super friendly.",
    "Great ambiance and delicious noodles. Will be back!",
    "Everything was perfect, especially the dumplings.",
]

SAMPLE_NEUTRAL = [
    "Food was okay, nothing special.",
    "Average experience. The wait time was reasonable.",
    "It was fine. Portions could be bigger.",
    "Not bad, not great."
]

SAMPLE_NEGATIVE = [
    "The noodles were overcooked and bland.",
    "Service was slow and my order was wrong.",
    "Too noisy and the broth was cold.",
    "Disappointed with the quality this time.",
]


def main() -> None:
    random.seed(7)
    init_db()
    today = datetime.utcnow().date()
    start = today - timedelta(days=29)
    for day_offset in range(30):
        day = start + timedelta(days=day_offset)
        # Random number of reviews per day
        for _ in range(random.randint(2, 8)):
            bucket = random.choices(
                population=["positive", "neutral", "negative"],
                weights=[0.5, 0.3, 0.2],
                k=1,
            )[0]
            if bucket == "positive":
                text = random.choice(SAMPLE_POSITIVE)
            elif bucket == "neutral":
                text = random.choice(SAMPLE_NEUTRAL)
            else:
                text = random.choice(SAMPLE_NEGATIVE)

            created_at = datetime.combine(day, datetime.min.time()) + timedelta(
                minutes=random.randint(0, 60 * 12)
            )
            insert_review(text, bucket, created_at)

    print("Seeded 30 days of synthetic reviews into SQLite.")


if __name__ == "__main__":
    main()

