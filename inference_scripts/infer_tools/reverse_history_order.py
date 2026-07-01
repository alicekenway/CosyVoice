#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


JsonRecord = Dict[str, Any]


def load_records(input_json: Path) -> List[JsonRecord]:
    with input_json.open("r", encoding="utf-8") as input_file:
        try:
            records = json.load(input_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {input_json}: {exc}") from exc

    if not isinstance(records, list):
        raise ValueError(f"Expected top-level JSON list in {input_json}")

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Expected object at {input_json}[{record_index}]")

    return records


def reverse_history(records: List[JsonRecord]) -> int:
    reversed_count = 0
    for record in records:
        history = record.get("history")
        if isinstance(history, list):
            record["history"] = list(reversed(history))
            reversed_count += 1
    return reversed_count


def write_records(output_json: Path, records: List[JsonRecord], indent: int | None) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=indent)
        output_file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reverse each JSON record's history list from newest-to-oldest "
            "into oldest-to-newest order."
        )
    )
    parser.add_argument(
        "--input-json",
        required=True,
        type=Path,
        help="Input JSON file containing a top-level list of objects.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Output JSON file with each history list reversed.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation for output. Use 0 for compact one-line JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_json = args.input_json.expanduser()
    output_json = args.output_json.expanduser()
    indent = None if args.indent == 0 else args.indent

    records = load_records(input_json)
    reversed_count = reverse_history(records)
    write_records(output_json, records, indent)

    print(f"Records: {len(records)}")
    print(f"Histories reversed: {reversed_count}")
    print(f"Output JSON: {output_json}")


if __name__ == "__main__":
    main()
