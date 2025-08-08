from __future__ import annotations

import os
from typing import Any, Tuple, Generator
import time
import threading

import gradio as gr
from dotenv import load_dotenv, find_dotenv

from .data.db import init_db
from .agents.response_agent import analyze_and_respond
from .agents.plot_agent import generate_sentiment_plot_from_prompt


def _bootstrap() -> None:
    # Robustly load .env from project root or parent dirs
    try:
        env_path = find_dotenv(usecwd=True)
        load_dotenv(env_path)
    except Exception:
        # Continue without raising; app can still run with fallback behavior
        pass
    init_db()


def ui_feedback_response(feedback_text: str) -> Tuple[str, str, str]:
    t0 = time.perf_counter()
    sentiment, reply = analyze_and_respond(feedback_text)
    elapsed = time.perf_counter() - t0
    runtime_html = f"<span style='font-size:12px;color:#64748b'>⏱ {elapsed:.1f}s</span>"
    return sentiment.capitalize(), reply, runtime_html


def ui_sentiment_plot_stream(prompt: str, chart_type: str) -> Generator[Tuple[Any, Any, str], None, None]:
    """Stream-only runtime timer without max cap while generating the plot.

    Yields (image, table, runtime_text) repeatedly. During processing, image and
    table are None and only runtime_text updates like "2.3s". On completion,
    yields the final image/table and the final elapsed seconds.
    """
    override = chart_type.lower() if chart_type else None
    result: dict[str, Any] = {}
    done = threading.Event()

    def worker():
        image, df = generate_sentiment_plot_from_prompt(prompt, override)
        result["image"] = image
        result["df"] = df
        done.set()

    t0 = time.perf_counter()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while not done.is_set():
        elapsed = time.perf_counter() - t0
        yield None, None, f"<span style='font-size:12px;color:#64748b'>⏱ {elapsed:.1f}s</span>"
        time.sleep(0.1)

    elapsed = time.perf_counter() - t0
    yield result.get("image"), result.get("df"), f"<span style='font-size:12px;color:#64748b'>⏱ {elapsed:.1f}s</span>"


def build_ui() -> gr.Blocks:
    runtime_css = """
    .runtime-tiny { width: auto !important; flex: 0 0 auto !important; margin-left: 8px; align-self: center; }
    .runtime-tiny p { margin: 0 !important; }
    """
    with gr.Blocks(theme=gr.themes.Soft(), css=runtime_css) as demo:
        gr.Markdown("""
        ### SteamNoodles — Automated Feedback Agents
        - Use the tabs below to analyze a single review or visualize sentiment trends.
        """)

        with gr.Tab("Feedback Response"):
            inp = gr.Textbox(label="Customer Review", placeholder="Type or paste a customer review…", lines=6)
            btn = gr.Button("Analyze & Reply")
            with gr.Row():
                out_sent = gr.Label(label="Detected Sentiment")
                sent_runtime = gr.HTML(value="", elem_classes=["runtime-tiny"]) 
            out_reply = gr.Textbox(label="Automated Reply", lines=4)
            btn.click(ui_feedback_response, inputs=[inp], outputs=[out_sent, out_reply, sent_runtime], show_progress="hidden")

        with gr.Tab("Sentiment Trends"):
            rng = gr.Textbox(
                label="Date Range Prompt",
                placeholder="e.g., last 7 days, June 1 to June 15, yesterday to today",
                value="last 7 days",
            )
            chart_choice = gr.Radio(["Auto", "Bar", "Line"], value="Auto", label="Chart Type")
            btn2 = gr.Button("Generate Plot")
            with gr.Row():
                img = gr.Image(label="Plot", type="pil")
                runtime = gr.HTML(value="", elem_classes=["runtime-tiny"]) 
            table = gr.Dataframe(label="Counts by Day", interactive=False)

            def _normalize_choice(choice: str) -> str:
                if choice.lower() == "bar":
                    return "bar"
                if choice.lower() == "line":
                    return "line"
                return ""  # Auto

            # Streaming wrapper must itself be a generator function (contain yield)
            def _plot_stream_wrapper(prompt_text: str, choice_text: str):
                normalized = _normalize_choice(choice_text)
                yield from ui_sentiment_plot_stream(prompt_text, normalized)

            btn2.click(
                fn=_plot_stream_wrapper,
                inputs=[rng, chart_choice],
                outputs=[img, table, runtime],
                show_progress="hidden",
            )

    return demo


def main() -> None:
    _bootstrap()
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()

