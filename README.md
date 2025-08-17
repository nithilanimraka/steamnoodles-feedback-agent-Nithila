## SteamNoodles — Automated Restaurant Feedback Agents

Two LangChain agents for customer feedback:
- Customer Feedback Response Agent
- Sentiment Visualization Agent

### Features
- Classify sentiment (positive/neutral/negative) and generate short, polite, context-aware replies
- Store reviews in SQLite for analytics
- Plot daily sentiment trends for a given date range (bar or line)
- Gradio web UI with two tabs

### Setup
1. Ensure Python 3.10+
2. Create and activate a virtual environment
   - macOS/Linux:
     - `python3 -m venv .venv && source .venv/bin/activate`
   - Windows (Powershell):
     - `py -m venv .venv; .venv\\Scripts\\Activate.ps1`
3. Install dependencies
   - `pip install -r requirements.txt`
4. Configure an LLM:
   - Create a `.env` file in project root with :
     ```
     OPENAI_API_KEY=sk-...your-key...
     ```

Without an LLM configured, the app will fall back to a rule-based sentiment model (VADER) and templated replies for demo purposes. The primary path uses LangChain with an LLM.

### Run the app
```
python -m app.main
```
Open the URL Gradio prints (default `http://127.0.0.1:7860`).

### Seed demo data (optional)
```
python -m scripts.seed_data
```
This populates the local SQLite DB with synthetic reviews over the last 30 days so you can test the plotting agent immediately.

### Project Structure
```
app/
  agents/
    llm_provider.py
    response_agent.py
    plot_agent.py
  data/
    db.py
  utils/
    date_utils.py
  main.py
scripts/
  seed_data.py
requirements.txt
README.md
```

### Notes
- The agents are implemented with LangChain. The response agent uses a JSON output schema to return both `sentiment` and `reply`. The plotting agent parses date ranges, queries the SQLite store, and generates bar/line charts per the user prompt.
- You can swap models by editing `app/agents/llm_provider.py`.
