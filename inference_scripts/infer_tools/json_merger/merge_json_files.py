#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_NAME = "merged.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple JSON files into one JSON file. Input files are "
            "provided as one colon-separated string, such as file1.json:file2.json."
        )
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help=(
            "Colon-separated input JSON files. Duplicate entries are allowed "
            "and are processed each time they appear."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the merged JSON file will be written.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output file name inside --output-dir. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "array", "object", "list"),
        default="auto",
        help=(
            "Merge mode. auto concatenates top-level arrays, recursively merges "
            "top-level objects, and wraps mixed JSON values in a list. array "
            "requires every file to contain a top-level array. object requires "
            "every file to contain a top-level object. list keeps each file's "
            "whole JSON value as one item in the output list."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Pretty-print indentation. Use -1 for compact JSON. Default: 2",
    )
    return parser.parse_args()


def parse_input_paths(inputs: str) -> list[Path]:
    raw_paths = inputs.split(":")
    if any(path == "" for path in raw_paths):
        raise ValueError("Empty input path found in --inputs")
    return [Path(path).expanduser() for path in raw_paths]


def read_json_file(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Input file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc


def detect_mode(values: list[Any]) -> str:
    if all(isinstance(value, list) for value in values):
        return "array"
    if all(isinstance(value, dict) for value in values):
        return "object"
    return "list"


def require_mode(values: list[Any], mode: str) -> None:
    expected_type = list if mode == "array" else dict
    for index, value in enumerate(values, start=1):
        if not isinstance(value, expected_type):
            actual_type = type(value).__name__
            raise ValueError(
                f"--mode {mode} requires input #{index} to be a JSON {mode}; "
                f"got {actual_type}"
            )


def merge_dicts(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for key, value in incoming.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merge_dicts(current, value)
        else:
            base[key] = value
    return base


def merge_values(values: list[Any], mode: str) -> tuple[Any, str]:
    effective_mode = detect_mode(values) if mode == "auto" else mode

    if effective_mode == "array":
        require_mode(values, "array")
        merged: list[Any] = []
        for value in values:
            merged.extend(value)
        return merged, effective_mode

    if effective_mode == "object":
        require_mode(values, "object")
        merged_object: dict[str, Any] = {}
        for value in values:
            merge_dicts(merged_object, value)
        return merged_object, effective_mode

    return values, effective_mode


def write_json_file(path: Path, value: Any, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_kwargs: dict[str, Any] = {"ensure_ascii": False}
    if indent >= 0:
        json_kwargs["indent"] = indent

    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(value, file, **json_kwargs)
            file.write("\n")
    except OSError as exc:
        raise ValueError(f"Could not write {path}: {exc}") from exc


def main() -> int:
    args = parse_args()

    try:
        input_paths = parse_input_paths(args.inputs)
        values = [read_json_file(path) for path in input_paths]
        merged, effective_mode = merge_values(values, args.mode)
        output_path = Path(args.output_dir).expanduser() / args.output_name
        write_json_file(output_path, merged, args.indent)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    unique_inputs = len({str(path) for path in input_paths})
    print(
        f"Merged {len(input_paths)} input occurrence(s) "
        f"from {unique_inputs} unique file(s) using {effective_mode} mode."
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
