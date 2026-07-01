#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

"""
Build v4 CosyVoice JSON input.

./prepare_cosyvoice_tsv.py \
  --text-file /path/to/text.txt \
  --audio-jsonl /path/to/audio.jsonl \
  --references-per-text 3 \
  --output-json /path/to/input.json
"""


JsonRecord = Dict[str, Any]


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


def _clean_text(value: Any) -> str:
    return str(value).strip()


def _record_id(record: JsonRecord, record_index: int) -> str:
    record_id = _clean_text(record.get("id", ""))
    if record_id:
        return record_id
    return f"record_{record_index:06d}"


def _texts_from_text_field(
    record: JsonRecord,
    json_path: Path,
    record_index: int,
) -> List[str]:
    text_value = record.get("text")
    if text_value is None:
        return []
    if not isinstance(text_value, list):
        raise ValueError(f"Expected text list at {json_path}[{record_index}]")
    return [_clean_text(text) for text in text_value if _clean_text(text)]


def _texts_from_input_history(
    record: JsonRecord,
    json_path: Path,
    record_index: int,
) -> List[str]:
    texts: List[str] = []

    history = record.get("history", [])
    if history is not None and not isinstance(history, list):
        raise ValueError(f"Expected history list at {json_path}[{record_index}]")

    if history is not None:
        for history_round in history:
            if not isinstance(history_round, list):
                continue
            if not history_round:
                continue
            history_text = _clean_text(history_round[0])
            if history_text:
                texts.append(history_text)

    input_text = _clean_text(record.get("input", ""))
    if input_text:
        texts.append(input_text)

    return texts


def load_json_records(json_path: Path) -> List[JsonRecord]:
    with json_path.open("r", encoding="utf-8") as json_file:
        try:
            records = json.load(json_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc

    if not isinstance(records, list):
        raise ValueError(f"Expected top-level JSON list in {json_path}")

    output_records: List[JsonRecord] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Expected object at {json_path}[{record_index}]")

        texts = _texts_from_text_field(record, json_path, record_index)
        if not texts:
            texts = _texts_from_input_history(record, json_path, record_index)
        if not texts:
            continue

        output_records.append(
            {
                "id": _record_id(record, record_index),
                "text": texts,
                "reference_audio_path": [],
            }
        )

    if not output_records:
        raise ValueError(f"No text found in {json_path}")
    return output_records


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


def build_text_file_records(texts: List[str], id_prefix: str) -> List[JsonRecord]:
    return [
        {
            "id": f"{id_prefix}_{text_index:06d}",
            "text": [text],
            "reference_audio_path": [],
        }
        for text_index, text in enumerate(texts)
    ]


def attach_reference_audio(
    records: List[JsonRecord],
    audio_paths: List[str],
    references_per_text: int,
) -> int:
    if references_per_text <= 0:
        raise ValueError("--references-per-text must be greater than 0")

    required_audio_count = len(records) * references_per_text
    if len(audio_paths) < required_audio_count:
        raise ValueError(
            f"Need {required_audio_count} reference audio paths for "
            f"{len(records)} records with --references-per-text "
            f"{references_per_text}, but only found {len(audio_paths)}"
        )

    for record_index, record in enumerate(records):
        start = record_index * references_per_text
        end = start + references_per_text
        record["reference_audio_path"] = audio_paths[start:end]

    return required_audio_count


def write_json(output_json: Path, records: List[JsonRecord]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build v4 CosyVoice JSON input from a text/JSON file and a JSONL "
            "file containing candidate reference audio paths."
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
            "JSON list file. If a record has a text list, use it directly. "
            "Otherwise extract input and the first element of each history round."
        ),
    )
    parser.add_argument(
        "--audio-jsonl",
        required=True,
        type=Path,
        help="JSONL file. Each line should contain an audio_filepath value.",
    )
    parser.add_argument(
        "--output-json",
        "--output-tsv",
        dest="output_json",
        required=True,
        type=Path,
        help=(
            "Output v4 JSON path. --output-tsv is accepted as a deprecated "
            "alias for older command lines."
        ),
    )
    parser.add_argument(
        "--audio-key",
        default="audio_filepath",
        help="JSONL key to use as the reference audio path (default: audio_filepath).",
    )
    parser.add_argument(
        "--text-id-prefix",
        default="text",
        help="ID prefix for records created from --text-file (default: text).",
    )
    parser.add_argument(
        "--references-per-text",
        "--num-references",
        type=int,
        default=3,
        help=(
            "Number of reference audios assigned to each prepared record, "
            "taken sequentially from --audio-jsonl (default: 3)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.json_file:
        records = load_json_records(args.json_file.expanduser())
    else:
        texts = load_texts(args.text_file.expanduser())
        records = build_text_file_records(texts, args.text_id_prefix)

    audio_paths = load_audio_paths(args.audio_jsonl.expanduser(), args.audio_key)
    used_audio_count = attach_reference_audio(
        records,
        audio_paths,
        args.references_per_text,
    )
    write_json(args.output_json.expanduser(), records)

    print(f"Records: {len(records)}")
    print(f"Text turns: {sum(len(record['text']) for record in records)}")
    print(f"References per record: {args.references_per_text}")
    print(f"Reference audio paths used: {used_audio_count}")
    if len(audio_paths) > used_audio_count:
        print(f"Reference audio paths unused: {len(audio_paths) - used_audio_count}")
    print(f"Output JSON: {args.output_json.expanduser()}")


if __name__ == "__main__":
    main()
