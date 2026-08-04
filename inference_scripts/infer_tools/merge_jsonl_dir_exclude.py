#!/usr/bin/env python3
import argparse
import os
import sys
import json
from pathlib import Path
from typing import Iterable, Set, Optional


def iter_jsonl_files(root: Path, recursive: bool, suffix: str = ".jsonl") -> Iterable[Path]:
    if recursive:
        for p in root.rglob(f"*{suffix}"):
            if p.is_file():
                yield p
    else:
        for p in root.glob(f"*{suffix}"):
            if p.is_file():
                yield p


def extract_audio_path(record: dict) -> Optional[str]:
    if not isinstance(record, dict):
        return None
    # Prefer common keys
    if "audio_filepath" in record and isinstance(record["audio_filepath"], str):
        return record["audio_filepath"]
    if "audio" in record and isinstance(record["audio"], str):
        return record["audio"]
    return None


def normalize_exclude_basename(audio_path: str) -> str:
    name = os.path.basename(audio_path)
    stem, ext = os.path.splitext(name)
    if stem.endswith("_16000"):
        stem = stem[: -len("_16000")]
    return f"{stem}{ext}" if ext else stem


def load_exclude_set(exclude_file: Path, encoding: str = "utf-8") -> Set[str]:
    exclude: Set[str] = set()
    if not exclude_file:
        return exclude
    if not exclude_file.exists():
        print(f"Warning: exclude file not found: {exclude_file}", file=sys.stderr)
        return exclude
    with exclude_file.open("r", encoding=encoding, errors="ignore") as fin:
        for raw in fin:
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            apath = extract_audio_path(obj)
            if not apath:
                continue
            exclude.add(normalize_exclude_basename(apath))
    return exclude


def merge_dir(
    input_dir: Path,
    output_path: Path,
    exclude_file: Path | None,
    recursive: bool,
    strip_ws: bool,
    encoding: str = "utf-8",
) -> tuple[int, int, int]:
    total_read = 0
    written = 0

    exclude_set = load_exclude_set(exclude_file) if exclude_file else set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding=encoding) as fout:
        for jsonl_path in iter_jsonl_files(input_dir, recursive=recursive):
            try:
                with jsonl_path.open("r", encoding=encoding) as fin:
                    for raw in fin:
                        if not raw.strip():
                            continue
                        total_read += 1
                        line = raw.rstrip("\n")
                        if strip_ws:
                            line = line.strip()
                        if not line:
                            continue
                        # Try JSON parse to extract audio path for comparison
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            # If not valid JSON, we cannot compare by audio path; keep the line
                            fout.write(raw if raw.endswith("\n") else raw + "\n")
                            written += 1
                            continue
                        apath = extract_audio_path(obj)
                        if apath:
                            base = os.path.basename(apath)
                            if base in exclude_set:
                                continue
                        fout.write(raw if raw.endswith("\n") else raw + "\n")
                        written += 1
            except FileNotFoundError:
                print(f"Warning: file not found during scan: {jsonl_path}", file=sys.stderr)
                continue
            except UnicodeDecodeError:
                # Fallback to ignoring errors
                with jsonl_path.open("r", encoding=encoding, errors="ignore") as fin:
                    for raw in fin:
                        if not raw.strip():
                            continue
                        total_read += 1
                        line = raw.rstrip("\n")
                        if strip_ws:
                            line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            fout.write(raw if raw.endswith("\n") else raw + "\n")
                            written += 1
                            continue
                        apath = extract_audio_path(obj)
                        if apath:
                            base = os.path.basename(apath)
                            if base in exclude_set:
                                continue
                        fout.write(raw if raw.endswith("\n") else raw + "\n")
                        written += 1

    return total_read, written, len(exclude_set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all .jsonl files from a directory into one file, excluding lines "
            "present in a given JSONL file. Skips empty lines."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing JSONL files")
    parser.add_argument("output", type=Path, help="Output JSONL file path")
    parser.add_argument(
        "--exclude",
        type=Path,
        default=None,
        help="JSONL file whose lines should be excluded from output",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search subdirectories (default: recursive)",
    )
    parser.add_argument(
        "--no-strip",
        action="store_true",
        help="Do not strip leading/trailing whitespace before comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"Input directory not found or not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(2)

    recursive = not args.no_recursive
    strip_ws = not args.no_strip

    total_read, written, excluded_unique = merge_dir(
        input_dir=args.input_dir,
        output_path=args.output,
        exclude_file=args.exclude,
        recursive=recursive,
        strip_ws=strip_ws,
    )

    print(
        f"Merged directory {args.input_dir} -> {args.output}. Read={total_read}, Written={written}, ExcludeSet={excluded_unique}",
        file=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover
    main()


