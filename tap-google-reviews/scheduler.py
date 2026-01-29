"""Minimal scheduler for the Reputation Guardian Agent.

Runs the Meltano tap pipeline every 12 hours.
Singer state is managed by Meltano for incremental sync.

Usage:
    python scheduler.py              # Loop forever, run every 12h
    python scheduler.py --once       # Run once and exit (for system cron)

System cron alternative:
    0 */12 * * * cd /path/to/tap-google-reviews && meltano run tap-google-reviews target-jsonl
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

INTERVAL_HOURS = 12


def run_pipeline() -> int:
    """Execute the Meltano tap -> target pipeline."""
    print(f"[{datetime.now().isoformat()}] Starting pipeline run...")
    result = subprocess.run(
        ["meltano", "run", "tap-google-reviews", "target-jsonl"],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"Pipeline failed: {result.stderr}", file=sys.stderr)
    else:
        print(f"[{datetime.now().isoformat()}] Pipeline completed successfully.")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Reputation Guardian Agent scheduler")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    if args.once:
        sys.exit(run_pipeline())

    while True:
        run_pipeline()
        print(f"Sleeping {INTERVAL_HOURS}h until next run...")
        time.sleep(INTERVAL_HOURS * 3600)


if __name__ == "__main__":
    main()
