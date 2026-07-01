#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy audio files from a JSONL manifest into output_dir/wav and "
            "write a new JSONL manifest with updated audio_filepath values."
        )
    )
    parser.add_argument("--input", required=True, help="Input JSONL manifest")
    parser.add_argument("--output-dir", required=True, help="Directory for wav/ and new JSONL")
    parser.add_argument(
        "--jsonl-name",
        help="Optional output JSONL filename. Defaults to the input JSONL filename.",
    )
    return parser.parse_args()


def output_manifest_path(input_path: str, output_dir: Path, jsonl_name: str | None) -> Path:
    name = jsonl_name or Path(input_path).name
    path = Path(name)
    if path.name != name:
        raise ValueError("--jsonl-name must be a filename, not a path")
    return output_dir / name


def unique_destination(wav_dir: Path, source_path: Path, used_names: set[str]) -> Path:
    suffix = source_path.suffix
    stem = source_path.stem
    candidate_name = source_path.name
    index = 1

    while candidate_name in used_names or (wav_dir / candidate_name).exists():
        candidate_name = f"{stem}_{index}{suffix}"
        index += 1

    used_names.add(candidate_name)
    return wav_dir / candidate_name


def rewrite_manifest(input_path: str, output_dir: str, jsonl_name: str | None) -> tuple[int, Path, Path]:
    output_root = Path(output_dir).expanduser()
    wav_dir = output_root / "wav"
    manifest_path = output_manifest_path(input_path, output_root, jsonl_name)

    output_root.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    written = 0

    with open(input_path, "r", encoding="utf-8") as in_f, open(
        manifest_path, "w", encoding="utf-8"
    ) as out_f:
        for line_number, line in enumerate(in_f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            audio_filepath = record.get("audio_filepath")
            if not isinstance(audio_filepath, str) or not audio_filepath:
                raise ValueError(
                    f"Line {line_number} does not have a non-empty string audio_filepath"
                )

            source_path = Path(audio_filepath).expanduser()
            if not source_path.is_file():
                raise FileNotFoundError(f"Audio file not found on line {line_number}: {source_path}")

            destination_path = unique_destination(wav_dir, source_path, used_names)
            shutil.copy2(source_path, destination_path)

            record["audio_filepath"] = str(destination_path.resolve())
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return written, wav_dir, manifest_path


def main() -> int:
    args = parse_args()

    try:
        written, wav_dir, manifest_path = rewrite_manifest(
            args.input,
            args.output_dir,
            args.jsonl_name,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Copied {written} audio files to {wav_dir}.")
    print(f"Wrote updated JSONL to {manifest_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
