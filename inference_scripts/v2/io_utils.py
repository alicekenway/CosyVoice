import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, TypeVar

from batch_types import TsvInputRow

LANG_TOKEN_MAP = {
    "zh": "<|zh|>",
    "en": "<|en|>",
    "ja": "<|ja|>",
    "yue": "<|yue|>",
    "ko": "<|ko|>",
}

_T = TypeVar("_T")


def normalize_header(header_name: str) -> str:
    return " ".join(header_name.strip().lower().replace("_", " ").split())


def resolve_columns(fieldnames: Sequence[str]) -> Dict[str, str]:
    column_mapping: Dict[str, str] = {}
    for column_name in fieldnames:
        normalized_name = normalize_header(column_name)
        if normalized_name == "text":
            column_mapping["text"] = column_name
        if normalized_name in {
            "reference audio path",
            "reference_audio_path",
            "reference wav path",
            "reference_wav_path",
            "ref audio path",
            "ref wav path",
            "audio path",
            "wav path",
            "prompt wav",
            "prompt wav path",
            "prompt speech path",
        }:
            column_mapping["ref_audio"] = column_name
    return column_mapping


def load_rows(input_tsv_path: Path) -> List[TsvInputRow]:
    rows: List[TsvInputRow] = []
    with input_tsv_path.open("r", encoding="utf-8", newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Input TSV has no header: {input_tsv_path}")
        column_mapping = resolve_columns(reader.fieldnames)
        if "text" not in column_mapping or "ref_audio" not in column_mapping:
            raise ValueError(
                "Input TSV must contain columns for text and reference audio path. "
                f"Found header: {reader.fieldnames}"
            )
        output_index = 0
        for row_index, row in enumerate(reader, start=1):
            text = (row.get(column_mapping["text"], "") or "").strip()
            ref_audio_path = (row.get(column_mapping["ref_audio"], "") or "").strip()
            if not text or not ref_audio_path:
                continue
            rows.append(
                TsvInputRow(
                    output_index=output_index,
                    row_id=str(row_index),
                    text=text,
                    ref_audio_path=ref_audio_path,
                )
            )
            output_index += 1
    return rows


def chunked(items: Sequence[_T], chunk_size: int) -> Iterator[Sequence[_T]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]


def write_metadata(output_tsv_path: Path, metadata_rows: Iterable[Dict[str, str]]) -> None:
    with output_tsv_path.open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=["speechpath", "text"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(metadata_rows)


def write_failures(failure_tsv_path: Path, failure_rows: Iterable[Dict[str, str]]) -> None:
    with failure_tsv_path.open("w", encoding="utf-8", newline="") as failure_file:
        writer = csv.DictWriter(
            failure_file,
            fieldnames=["row_id", "text", "ref_audio", "error"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(failure_rows)
