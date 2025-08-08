from __future__ import annotations

import io
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from ..data.db import get_sentiment_counts_by_day
from ..utils.date_utils import parse_date_range, human_range_label
from .llm_provider import get_llm


ChartType = Literal["bar", "line"]


class PlotDecision(BaseModel):
    chart: ChartType = Field(description="Chart type (bar or line)")
    title: str = Field(description="Short, human-friendly chart title")


def _decide_chart_type(user_prompt: str, override: Optional[str] = None) -> Tuple[ChartType, str]:
    if override in {"bar", "line"}:
        chart = override  # type: ignore
    else:
        llm = get_llm()
        if llm is None:
            # Heuristic fallback
            chart = "bar" if "bar" in user_prompt.lower() else "line"
            title = "Sentiment trend"
            return chart, title

        parser = JsonOutputParser(pydantic_object=PlotDecision)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a data visualization assistant. Given a user prompt about sentiment trends, "
                    "choose 'bar' or 'line' for the chart and produce a short title. Return JSON only.",
                ),
                ("human", "{prompt}"),
            ]
        )
        chain = prompt | llm | parser
        try:
            result = chain.invoke({"prompt": user_prompt})
            if isinstance(result, dict):
                decision = PlotDecision(**result)
            else:
                decision = result  # type: ignore
            return decision.chart, decision.title
        except Exception:
            chart = "bar" if "bar" in user_prompt.lower() else "line"
            return chart, "Sentiment trend"

    return chart, "Sentiment trend"


def _plot_counts(df: pd.DataFrame, chart: ChartType, title: str, subtitle: str) -> Image.Image:
    if df.empty:
        # Create an empty image with a message
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data for selected range", ha="center", va="center")
        ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        if chart == "bar":
            width = 0.25
            x = range(len(dates))
            ax.bar([i - width for i in x], df["positive"], width=width, label="Positive", color="#4CAF50")
            ax.bar(x, df["neutral"], width=width, label="Neutral", color="#FFC107")
            ax.bar([i + width for i in x], df["negative"], width=width, label="Negative", color="#F44336")
            ax.set_xticks(list(x))
            ax.set_xticklabels(dates, rotation=45, ha="right")
        else:
            ax.plot(dates, df["positive"], marker="o", label="Positive", color="#4CAF50")
            ax.plot(dates, df["neutral"], marker="o", label="Neutral", color="#FFC107")
            ax.plot(dates, df["negative"], marker="o", label="Negative", color="#F44336")
            plt.xticks(rotation=45, ha="right")

        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count of Reviews")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def generate_sentiment_plot_from_prompt(user_prompt: str, chart_override: Optional[str] = None):
    start_date, end_date = parse_date_range(user_prompt)
    chart, title = _decide_chart_type(user_prompt, chart_override)
    df = get_sentiment_counts_by_day(start_date, end_date)
    subtitle = human_range_label(start_date, end_date)
    image = _plot_counts(df, chart, title, subtitle)
    return image, df

