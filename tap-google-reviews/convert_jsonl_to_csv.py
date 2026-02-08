import json
import csv
import sys
from pathlib import Path

DEFAULT_INPUT = "output.jsonl"
OUTPUT_FILE = "reviews.csv"


def jsonl_to_csv(input_path: str, output_path: str):
    rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            message = json.loads(line)

            # Handle both Singer RECORD messages and raw records (from target-jsonl)
            if "type" in message:
                # Singer format: {"type": "RECORD", "stream": "...", "record": {...}}
                if message["type"] != "RECORD":
                    continue
                record = message.get("record", {})
            else:
                # Raw record format (from Meltano target-jsonl)
                record = message

            rows.append(record)

    if not rows:
        print("No records found. CSV not created.")
        return

    # Use keys from first record as CSV headers
    fieldnames = rows[0].keys()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Converted {len(rows)} records -> {output_path}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    jsonl_to_csv(input_file, OUTPUT_FILE)
