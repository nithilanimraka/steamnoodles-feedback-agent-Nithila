from __future__ import annotations

from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .llm_provider import get_llm, llm_available
from ..data.db import insert_review


SentimentLabel = Literal["positive", "neutral", "negative"]


class ResponseSchema(BaseModel):
    sentiment: SentimentLabel = Field(description="The sentiment of the review")
    reply: str = Field(description="Short, polite, context-aware reply to the customer")


def _rule_based_sentiment(text: str) -> SentimentLabel:
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.2:
        return "positive"
    if score <= -0.2:
        return "negative"
    return "neutral"


def _templated_reply(text: str, sentiment: SentimentLabel) -> str:
    if sentiment == "positive":
        return (
            "Thanks for the wonderful feedback! We're thrilled you enjoyed your experience at SteamNoodles. "
            "We hope to welcome you back soon!"
        )
    if sentiment == "negative":
        return (
            "We're sorry to hear about your experience. Thank you for letting us know—our team will review this "
            "and work to make things right. We hope you'll give us another chance."
        )
    return (
        "Thank you for sharing your thoughts. We appreciate your feedback and will use it to keep improving."
    )


def analyze_and_respond(feedback_text: str) -> Tuple[SentimentLabel, str]:
    """Analyze sentiment and generate an automated reply. Saves the review to DB.

    Uses an LLM via LangChain when available. Falls back to a local rule-based
    classifier and templated replies for offline demo.
    """
    text = feedback_text.strip()
    if not text:
        raise ValueError("Feedback text is empty.")

    if llm_available():
        llm = get_llm()
        parser = JsonOutputParser(pydantic_object=ResponseSchema)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful customer support agent for SteamNoodles (a modern restaurant). "
                    "Determine the sentiment as one of: positive, neutral, negative. "
                    "Then craft a short, polite, context-aware reply. Be concise (<= 2 sentences). "
                    "Return JSON only following the provided schema.",
                ),
                ("human", "Customer review: {review}"),
            ]
        ).partial()
        chain = prompt | llm | parser
        try:
            result = chain.invoke({"review": text})
            # Ensure we have a ResponseSchema instance even if parser returns a dict
            if isinstance(result, dict):
                parsed = ResponseSchema(**result)
            else:
                parsed = result  # type: ignore
            sentiment = parsed.sentiment
            reply = parsed.reply.strip()
        except Exception:
            # Robust fallback if LLM or parsing fails
            sentiment = _rule_based_sentiment(text)
            reply = _templated_reply(text, sentiment)
    else:
        sentiment = _rule_based_sentiment(text)
        reply = _templated_reply(text, sentiment)

    insert_review(text, sentiment)
    return sentiment, reply

