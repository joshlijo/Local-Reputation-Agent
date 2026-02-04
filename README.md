# Local Reputation Agent

An agentic reputation management system for local businesses. Scrapes Google Maps reviews, runs sentiment analysis, and drafts AI-powered responses for negative reviews.

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  Google Maps        │     │  Sentiment          │     │  Agent              │
│  Scraper            │────▶│  Analysis           │────▶│  (Streamlit UI)     │
│  (tap-google-reviews)     │  Pipeline           │     │                     │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
    output.jsonl              reviews.csv               reputation.db
    (Singer format)           (processed)              (SQLite)
```

## Prerequisites

- Python 3.10+
- Playwright browsers installed
- (Optional) Google Gemini API key for AI response drafting

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

### 2. Configure the scraper

Edit `tap-google-reviews/config.json` with your business URL:

```json
{
  "google_maps_url": "https://www.google.com/maps/place/YOUR+BUSINESS+NAME/...",
  "headless": true,
  "max_pages": 100,
  "rate_limit_seconds": 1.0,
  "initial_full_scrape": true
}
```

### 3. Run the scraper

```bash
cd tap-google-reviews
python -m tap_google_reviews.tap --config config.json > output.jsonl
python convert_jsonl_to_csv.py
```

This creates `reviews.csv` with all extracted reviews.

### 4. Set up environment variables (optional)

For AI-powered response drafting, set your Gemini API key:

```bash
# Windows
set GOOGLE_API_KEY=your-api-key-here

# macOS/Linux
export GOOGLE_API_KEY=your-api-key-here
```

Other optional environment variables:
- `BUSINESS_NAME` - Your business name (default: "Cafe Amudham")
- `GEMINI_MODEL` - Model to use (default: "gemini-flash-lite-latest")
- `SCHEDULE_HOURS` - Hours between scheduled runs (default: 6)

### 5. Run the agent scheduler

```bash
cd ..
python agent/scheduler.py --once
```

This processes reviews through sentiment analysis and queues negative reviews for response.

### 6. Launch the dashboard

```bash
streamlit run agent/app.py
```

Open http://localhost:8501 to view:
- **Pulse tab**: Reputation score, sentiment breakdown, top complaints
- **Review Queue tab**: AI-drafted responses pending approval

## Project Structure

```
Local Reputation Agent/
├── tap-google-reviews/          # Google Maps review scraper
│   ├── tap_google_reviews/      # Singer SDK tap implementation
│   │   ├── scraper.py           # Playwright-based scraping logic
│   │   ├── tap.py               # Entry point
│   │   └── streams.py           # Singer stream definitions
│   ├── config.json              # Scraper configuration
│   └── convert_jsonl_to_csv.py  # JSONL to CSV converter
├── sentiment-analysis/          # Sentiment analysis pipeline
│   ├── sentiment.py             # VADER + rating-based classification
│   ├── aspects.py               # Aspect extraction (food, service, etc.)
│   └── urgency.py               # Urgency detection
├── agent/                       # Agentic UI and scheduling
│   ├── app.py                   # Streamlit dashboard
│   ├── scheduler.py             # Background processing loop
│   ├── db.py                    # SQLite persistence
│   └── response_agent.py        # AI response drafting (Gemini)
└── requirements.txt             # Python dependencies
```

## Pipeline Flow

1. **Scrape** → `tap-google-reviews` extracts reviews from Google Maps
2. **Convert** → `convert_jsonl_to_csv.py` transforms Singer output to CSV
3. **Process** → `agent/scheduler.py` runs sentiment analysis on new reviews
4. **Draft** → Negative reviews (rating ≤ 3) get AI-drafted responses
5. **Review** → Human approves/edits responses in Streamlit UI

## Scheduled Operation

For continuous monitoring, run the scheduler in loop mode:

```bash
python agent/scheduler.py
```

This runs every 6 hours (configurable via `SCHEDULE_HOURS`).

Alternatively, use system cron:

```bash
# Run every 12 hours
0 */12 * * * cd /path/to/project && python agent/scheduler.py --once
```

## Troubleshooting

### "Input CSV not found"

Run the scraper first to generate `reviews.csv`:

```bash
cd tap-google-reviews
python -m tap_google_reviews.tap --config config.json > output.jsonl
python convert_jsonl_to_csv.py
```

### Scraper returns zero reviews

- Check that `config.json` has the correct Google Maps URL
- Try setting `"headless": false` to debug visually
- Google Maps UI may have changed; check for scraper updates

### AI responses not generating

- Verify `GOOGLE_API_KEY` environment variable is set
- Check the Gemini API quota/limits
- Responses are capped at 5 per run (configurable in `scheduler.py`)
