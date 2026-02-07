# Local Reputation Agent

An agentic reputation management system for local businesses. Automatically scrapes Google Maps reviews, runs sentiment analysis, drafts AI-powered responses for negative reviews, and queues them for human approval.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        agent/scheduler.py                           │
│                     (The Heartbeat — every 6h)                      │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │ 1. Scrape│──▶│ 2. Detect│──▶│ 3. Analyze│──▶│ 4. Draft       │  │
│  │  Reviews │   │  New     │   │  Sentiment│   │  AI Responses  │  │
│  └──────────┘   └──────────┘   └──────────┘   └────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
         │                                               │
         ▼                                               ▼
   reviews.csv                                    reputation.db
                                                         │
                                                         ▼
                                              ┌────────────────┐
                                              │ 5. Human Review│
                                              │  (Streamlit UI)│
                                              └────────────────┘
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

- Python 3.10+
- Playwright browsers installed
- Google Gemini API key for AI response drafting

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd "Local Reputation Agent"

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
playwright install chromium
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your-api-key-here
BUSINESS_NAME=Your Business Name
```

Optional variables:
- `GEMINI_MODEL` — Model to use (default: `gemini-flash-lite-latest`)
- `SCHEDULE_HOURS` — Hours between scheduled runs (default: `6`)
- `MAX_DRAFTS_PER_RUN` — Max AI responses per cycle (default: `5`)

### 3. Configure the scraper

```bash
cp tap-google-reviews/config.json.example tap-google-reviews/config.json
```

Then edit `tap-google-reviews/config.json` with your business URL:

```json
{
  "google_maps_url": "https://maps.app.goo.gl/YOUR_BUSINESS_LINK",
  "headless": true,
  "max_pages": 100,
  "rate_limit_seconds": 1.5,
  "initial_full_scrape": true
}
```

To get the URL: open Google Maps, find your business, click **Share**, and copy the link.

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

## Project Structure

```
Local Reputation Agent/
├── agent/                        # Agentic core
│   ├── scheduler.py              # Orchestrator — runs tap + sentiment + drafting
│   ├── app.py                    # Streamlit dashboard (Pulse + Review Queue)
│   ├── db.py                     # SQLite persistence (reviews + response queue)
│   ├── response_agent.py         # AI response drafting via Google ADK + Gemini
│   └── agent_config.py           # Environment and config loading
├── tap-google-reviews/           # Google Maps review scraper
│   ├── tap_google_reviews/       # Singer SDK tap implementation
│   │   ├── scraper.py            # Playwright-based scraping with stealth
│   │   ├── tap.py                # Singer SDK entry point
│   │   └── streams.py            # Singer stream definitions
│   ├── config.json               # Scraper configuration
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

- Check that `config.json` has the correct `google_maps_url` for your business
- Try setting `"headless": false` to debug visually
- Google Maps may serve reduced UI to automated browsers; see stealth notes in the codebase

### AI responses not generating

- Verify `GEMINI_API_KEY` is set in `.env`
- Check the Gemini API quota (free tier: ~20 requests/day)
- Responses are capped at `MAX_DRAFTS_PER_RUN` per cycle (default: 5)

### "Tap failed" warning

The scheduler will log this if the scraper errors out, then fall back to processing whatever CSV already exists from a previous run. Check the scraper logs for details.
