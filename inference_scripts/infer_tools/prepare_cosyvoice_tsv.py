#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple
'''
./prepare_cosyvoice_tsv.py \
  --text-file /path/to/text.txt \
  --audio-jsonl /path/to/audio.jsonl \
  --output-tsv /path/to/output.tsv
'''

def load_texts(text_path: Path) -> List[str]:
    texts: List[str] = []
    with text_path.open("r", encoding="utf-8") as text_file:
        for raw_line in text_file:
            text = raw_line.rstrip("\n").strip()
            if text:
                texts.append(text)
    if not texts:
        raise ValueError(f"No non-empty text lines found in {text_path}")
    return texts


def load_json_texts(json_path: Path) -> List[Tuple[str, str]]:
    with json_path.open("r", encoding="utf-8") as json_file:
        try:
            records = json.load(json_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc

    if not isinstance(records, list):
        raise ValueError(f"Expected top-level JSON list in {json_path}")

    rows: List[Tuple[str, str]] = []
    for data_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Expected object at {json_path}[{data_index}]")

        input_text = str(record.get("input", "")).strip()
        if input_text:
            rows.append((f"input_{data_index}", input_text))

        history = record.get("history", [])
        if history is None:
            continue
        if not isinstance(history, list):
            raise ValueError(f"Expected history list at {json_path}[{data_index}]")

        for history_index, history_round in enumerate(history):
            if not isinstance(history_round, list):
                continue
            if not history_round:
                continue
            history_text = str(history_round[0]).strip()
            if history_text:
                rows.append((f"history_{data_index}_{history_index}", history_text))

    if not rows:
        raise ValueError(f"No input or history text found in {json_path}")
    return rows


def load_audio_paths(jsonl_path: Path, audio_key: str) -> List[str]:
    audio_paths: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, raw_line in enumerate(jsonl_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {jsonl_path}:{line_number}: {exc}"
                ) from exc

            audio_path = str(record.get(audio_key, "")).strip()
            if not audio_path:
                continue
            audio_paths.append(audio_path)

    if not audio_paths:
        raise ValueError(f"No '{audio_key}' values found in {jsonl_path}")
    return audio_paths


def write_tsv(
    output_tsv: Path,
    texts: List[str],
    audio_paths: List[str],
    ids: Optional[List[str]] = None,
) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text", "reference_audio_path"]
    if ids is not None:
        fieldnames.insert(0, "id")

    with output_tsv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for index, text in enumerate(texts):
            row = {
                "text": text,
                "reference_audio_path": audio_paths[index % len(audio_paths)],
            }
            if ids is not None:
                row["id"] = ids[index]
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CosyVoice TSV from a text/JSON file and a JSONL file containing "
            "reference audio paths."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text-file",
        type=Path,
        help="Text file with one TTS sentence per line.",
    )
    input_group.add_argument(
        "--json-file",
        type=Path,
        help=(
            "JSON list file. Extracts each record's input and the first element "
            "of each history round, and writes an id column."
        ),
    )
    parser.add_argument(
        "--audio-jsonl",
        required=True,
        type=Path,
        help="JSONL file. Each line should contain an audio_filepath value.",
    )
    parser.add_argument(
        "--output-tsv",
        required=True,
        type=Path,
        help="Output TSV path, compatible with CosyVoice TSV batch inference.",
    )
    parser.add_argument(
        "--audio-key",
        default="audio_filepath",
        help="JSONL key to use as the reference audio path (default: audio_filepath).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = None
    if args.json_file:
        id_text_rows = load_json_texts(args.json_file.expanduser())
        ids = [row_id for row_id, _ in id_text_rows]
        texts = [text for _, text in id_text_rows]
    else:
        texts = load_texts(args.text_file.expanduser())

    audio_paths = load_audio_paths(args.audio_jsonl.expanduser(), args.audio_key)
    write_tsv(args.output_tsv.expanduser(), texts, audio_paths, ids)

    print(f"Text rows: {len(texts)}")
    print(f"Reference audio paths: {len(audio_paths)}")
    print(f"Output TSV: {args.output_tsv.expanduser()}")


if __name__ == "__main__":
    main()
