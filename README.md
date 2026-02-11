# Local Reputation Agent

An agentic reputation management system for local businesses. Automatically scrapes Google Maps reviews, runs sentiment analysis, drafts AI-powered responses for negative reviews, and queues them for human approval.

## Architecture

```
                    ┌──────────────────────────┐
                    │      scheduler.py        │
                    │    (Heartbeat — 6h)      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  1. SCRAPE               │
                    │  Meltano Tap             │
                    │  (Playwright + Singer)   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  2. CONVERT & DIFF       │
                    │  JSONL → CSV → compare   │
                    │  against DB (new only)   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  3. ANALYZE              │
                    │  Sentiment + Aspects     │
                    │  + Urgency Detection     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  4. DRAFT (negatives)    │
                    │  Google ADK Agent        │
                    │  (Gemini API)            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      reputation.db       │
                    │  (reviews + draft queue) │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  5. HUMAN REVIEW         │
                    │  Streamlit UI            │
                    │  (Approve / Edit /       │
                    │   Reject drafts)         │
                    └──────────────────────────┘
```

## Full Workflow

1. **Heartbeat** — `scheduler.py` runs on a loop (default: every 6 hours)
2. **Scrape** — Triggers the Google Maps tap via Playwright to fetch fresh reviews
3. **Convert** — Transforms Singer JSONL output into CSV
4. **Diff** — Compares against SQLite DB to find only new reviews
5. **Analyze** — Runs sentiment, aspect, and urgency detection on each new review
6. **Draft** — Sends negative reviews (rating ≤ 3) to Gemini API; urgent reviews prioritized
7. **Queue** — Saves AI drafts to a "pending" review queue in SQLite
8. **Human Review** — Manager opens Streamlit UI, reads the draft, edits if needed, clicks Approve or Reject

Responses are **never auto-posted** to Google. Approved responses stay in the database for manual posting.

## Prerequisites

Works on **Windows, macOS, and Linux**.

- **Google Chrome (stable)** -> [Download here](https://www.google.com/chrome/)
- **Git** -> [Download here](https://git-scm.com/downloads) (or download the repo as a ZIP from GitHub)
- **Python 3.10+** -> [Download here](https://python.org/downloads). `pip` comes bundled with Python, no separate install needed.
  - macOS alternative: `brew install python`
  - Verify with: `python --version`
- **Google Gemini API key** — Get one free at [AI Studio](https://aistudio.google.com/)

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/joshlijo/Local-Reputation-Agent.git
```
Open the Local-Reputation-Agent folder on your IDE.
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
meltano install
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your-api-key-here
```

Optional variables:
- `GEMINI_MODEL` — Model to use (default: `gemini-flash-lite-latest`)
- `SCHEDULE_HOURS` — Hours between scheduled runs (default: `6`)
- `MAX_DRAFTS_PER_RUN` — Max AI responses per cycle (default: `5`)

### 3. Configure the scraper

To get your business URL: open Google Maps, find your business, click **Share**, and copy the link.

**a) Edit `meltano.yml`** — this is the primary config used by the Meltano ELT pipeline. Update the `config` section under `plugins > extractors > tap-google-reviews`:

```yaml
config:
  google_maps_url: YOUR_BUSINESS_LINK
  place_query: Your Business Name
  headless: false
  max_pages: 100
  rate_limit_seconds: 1.5
  initial_full_scrape: true
```

**b) Edit `tap-google-reviews/config.json`** — used as a fallback if Meltano encounters issues:

macOS / Linux / Git Bash
```bash
cp tap-google-reviews/config.json.example tap-google-reviews/config.json
```
Windows (Command Prompt)
```bash
copy tap-google-reviews\config.json.example tap-google-reviews\config.json
```

Then update it with the same values:

```json
{
  "google_maps_url": "YOUR_BUSINESS_LINK",
  "place_query": "Your Business Name",
  "headless": false,
  "max_pages": 100,
  "rate_limit_seconds": 1.5,
  "initial_full_scrape": true
}
```

> **Why two files?** The scheduler tries the Meltano pipeline first (`meltano run`) and falls back to direct tap invocation if Meltano encounters issues. Both paths need your business details.

### 4. Run the full pipeline (one-shot)

```bash
python agent/scheduler.py --once
```

This single command does everything: scrapes reviews, runs sentiment analysis, and drafts AI responses.

### 5. Launch the dashboard

```bash
streamlit run agent/app.py
```

Open http://localhost:8501 to view:
- **Pulse tab** — Reputation score, sentiment breakdown, top complaints this week, recent negative reviews
- **Review Queue tab** — AI-drafted responses pending human approval/editing

### 6. Continuous monitoring

For always-on operation, run the scheduler in loop mode:

```bash
python agent/scheduler.py
```

This repeats the full pipeline every 6 hours (configurable via `SCHEDULE_HOURS`).

Alternatively, use system cron:

```bash
# Run every 6 hours
0 */6 * * * cd /path/to/project && .venv/bin/python agent/scheduler.py --once
```

### Meltano ELT Pipeline

The scraper is built as a custom Singer-compliant tap (`tap-google-reviews`) using the [Meltano SDK](https://sdk.meltano.com/), following the pluggable ELT architecture. This means the data extraction is decoupled from loading — today it scrapes Google Maps, but the source can be swapped (e.g., Facebook API, CSV import) without breaking the rest of the system.

The scheduler automatically runs `meltano run tap-google-reviews target-jsonl` to execute the full ELT pipeline with Singer-spec state tracking. To run the pipeline manually:

```bash
meltano run tap-google-reviews target-jsonl
```

## Project Structure

```
Local Reputation Agent/
├── meltano.yml                   # Meltano ELT pipeline configuration
├── agent/                        # Agentic core
│   ├── scheduler.py              # Orchestrator — runs tap + sentiment + drafting
│   ├── app.py                    # Streamlit dashboard (Pulse + Review Queue)
│   ├── db.py                     # SQLite persistence (reviews + response queue)
│   ├── response_agent.py         # AI response drafting via Google ADK + Gemini
│   └── agent_config.py           # Environment and config loading
├── tap-google-reviews/           # Custom Singer tap (Meltano SDK)
│   ├── tap_google_reviews/       # Python package
│   │   ├── scraper.py            # Playwright-based scraping with stealth
│   │   ├── tap.py                # Singer SDK entry point
│   │   └── streams.py            # Singer stream definitions
│   ├── config.json               # Scraper configuration (fallback for direct invocation)
│   └── convert_jsonl_to_csv.py   # JSONL to CSV converter
├── sentiment-analysis/           # Sentiment analysis pipeline
│   ├── sentiment.py              # VADER + rating-based classification
│   ├── aspects.py                # Aspect extraction (food, service, etc.)
│   ├── urgency.py                # Urgency detection (legal threats, health risks)
│   └── sentiment_rules.py        # Contract enforcement / validation
├── .env                          # API keys (not committed)
└── requirements.txt              # Python dependencies
```

## Troubleshooting

### Scraper returns zero reviews

- Check that `meltano.yml` and `tap-google-reviews/config.json` have the correct `google_maps_url` and `place_query` for your business
- Try setting `"headless": false` to debug visually
- Google Maps may serve reduced UI to automated browsers; see stealth notes in the codebase

### AI responses not generating

- Verify `GEMINI_API_KEY` is set in `.env`
- Check the Gemini API quota (free tier limits vary by model; check [AI Studio](https://aistudio.google.com/) for your current limits)
- Responses are capped at `MAX_DRAFTS_PER_RUN` per cycle (default: 5)

### "Tap failed" warning

The scheduler will log this if the scraper errors out, then fall back to processing whatever CSV already exists from a previous run. Check the scraper logs for details.