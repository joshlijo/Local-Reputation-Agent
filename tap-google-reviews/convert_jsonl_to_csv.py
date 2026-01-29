import json
import csv
from pathlib import Path

INPUT_FILE = "output2.jsonl"
OUTPUT_FILE = "reviews.csv"

def jsonl_to_csv(input_path: str, output_path: str):
    rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            message = json.loads(line)

            # Only process Singer RECORD messages
            if message.get("type") != "RECORD":
                continue

            record = message.get("record", {})
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

    print(f"Converted {len(rows)} records → {output_path}")


if __name__ == "__main__":
    jsonl_to_csv(INPUT_FILE, OUTPUT_FILE)