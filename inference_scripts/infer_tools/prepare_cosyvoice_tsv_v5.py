#!/usr/bin/env python3
"""Build transcript-conditioned CosyVoice v5 JSON input."""

import argparse
from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Dict, List


JsonRecord = Dict[str, Any]


@dataclass(frozen=True)
class AudioReference:
    path: str
    text: str


def load_texts(text_path: Path) -> List[str]:
    texts = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines()]
    texts = [text for text in texts if text]
    if not texts:
        raise ValueError(f"No non-empty text lines found in {text_path}")
    return texts


def _clean_text(value: Any) -> str:
    return str(value).strip()


def _record_id(record: JsonRecord, record_index: int) -> str:
    return _clean_text(record.get("id", "")) or f"record_{record_index:06d}"


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
    for history_round in history or []:
        if isinstance(history_round, list) and history_round:
            history_text = _clean_text(history_round[0])
            if history_text:
                texts.append(history_text)
    input_text = _clean_text(record.get("input", ""))
    if input_text:
        texts.append(input_text)
    return texts


def _empty_record(record_id: str, texts: List[str]) -> JsonRecord:
    return {
        "id": record_id,
        "text": texts,
        "reference_audio_path": [],
        "reference_audio_text": [],
    }


def load_json_records(json_path: Path) -> List[JsonRecord]:
    try:
        records = json.loads(json_path.read_text(encoding="utf-8"))
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
        if texts:
            output_records.append(
                _empty_record(_record_id(record, record_index), texts)
            )
    if not output_records:
        raise ValueError(f"No text found in {json_path}")
    return output_records


def load_audio_references(
    jsonl_path: Path,
    audio_key: str,
    audio_text_key: str,
    audio_root: Path,
    limit: int | None = None,
) -> List[AudioReference]:
    references: List[AudioReference] = []
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
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {jsonl_path}:{line_number}")

            audio_value = _clean_text(record.get(audio_key, ""))
            if not audio_value:
                raise ValueError(
                    f"Missing or empty '{audio_key}' at {jsonl_path}:{line_number}"
                )
            reference_text = _clean_text(record.get(audio_text_key, ""))
            if not reference_text:
                raise ValueError(
                    f"Missing or empty '{audio_text_key}' at "
                    f"{jsonl_path}:{line_number}"
                )

            audio_path = Path(audio_value).expanduser()
            if not audio_path.is_absolute():
                audio_path = audio_root / audio_path
            references.append(
                AudioReference(
                    path=str(audio_path),
                    text=reference_text,
                )
            )
            if limit is not None and len(references) >= limit:
                break
    if not references:
        raise ValueError(f"No audio references found in {jsonl_path}")
    return references


def build_text_file_records(texts: List[str], id_prefix: str) -> List[JsonRecord]:
    return [
        _empty_record(f"{id_prefix}_{text_index:06d}", [text])
        for text_index, text in enumerate(texts)
    ]


def attach_references(
    records: List[JsonRecord],
    references: List[AudioReference],
    references_per_text: int,
    no_shuffle: bool = False,
    seed: int = 0,
) -> int:
    if references_per_text <= 0:
        raise ValueError("--references-per-text must be greater than 0")
    required_count = len(records) * references_per_text
    if len(references) < required_count:
        raise ValueError(
            f"Need {required_count} audio/transcript pairs for {len(records)} "
            f"records with --references-per-text {references_per_text}, but only "
            f"found {len(references)}"
        )
    selected = (
        references[:required_count]
        if no_shuffle
        else random.Random(seed).sample(references, required_count)
    )
    for record_index, record in enumerate(records):
        start = record_index * references_per_text
        end = start + references_per_text
        record_references = selected[start:end]
        record["reference_audio_path"] = [item.path for item in record_references]
        record["reference_audio_text"] = [item.text for item in record_references]
    return required_count


def write_json(output_json: Path, records: List[JsonRecord]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build CosyVoice v5 JSON input with aligned reference audio paths "
            "and transcripts."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text-file", type=Path, help="Text file with one TTS sentence per line."
    )
    input_group.add_argument(
        "--json-file",
        type=Path,
        help=(
            "JSON list. Use each text list directly, or extract input and the "
            "first element of each history round."
        ),
    )
    parser.add_argument("--audio-jsonl", required=True, type=Path)
    parser.add_argument(
        "--output-json",
        "--output-tsv",
        dest="output_json",
        required=True,
        type=Path,
        help="Output v5 JSON path; --output-tsv remains a deprecated alias.",
    )
    parser.add_argument("--audio-key", default="audio_filepath")
    parser.add_argument("--audio-text-key", default="text")
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="Base for relative audio paths (default: audio JSONL directory).",
    )
    parser.add_argument("--text-id-prefix", default="text")
    parser.add_argument(
        "--references-per-text", "--num-references", type=int, default=3
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Use JSONL rows sequentially instead of random sampling.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.json_file:
        records = load_json_records(args.json_file.expanduser().resolve())
    else:
        texts = load_texts(args.text_file.expanduser().resolve())
        records = build_text_file_records(texts, args.text_id_prefix)

    jsonl_path = args.audio_jsonl.expanduser().resolve()
    audio_root = (
        args.audio_root.expanduser().resolve()
        if args.audio_root
        else jsonl_path.parent
    )
    references = load_audio_references(
        jsonl_path,
        args.audio_key,
        args.audio_text_key,
        audio_root,
        limit=(len(records) * args.references_per_text if args.no_shuffle else None),
    )
    used_count = attach_references(
        records,
        references,
        args.references_per_text,
        no_shuffle=args.no_shuffle,
        seed=args.seed,
    )
    output_json = args.output_json.expanduser().resolve()
    write_json(output_json, records)

    print(f"Records: {len(records)}")
    print(f"Text turns: {sum(len(record['text']) for record in records)}")
    print(f"References per record: {args.references_per_text}")
    print(
        "Reference selection: "
        + (
            "sequential JSONL order"
            if args.no_shuffle
            else f"random sample without replacement (seed={args.seed})"
        )
    )
    print(f"Audio/transcript pairs used: {used_count}")
    print(f"Audio/transcript pairs unused: {len(references) - used_count}")
    print(f"Output JSON: {output_json}")


if __name__ == "__main__":
    main()
