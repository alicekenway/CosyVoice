import csv
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Sequence, TypeVar

if TYPE_CHECKING:
    from batch_types import TsvInputRow

LANG_TOKEN_MAP = {
    "zh": "<|zh|>",
    "en": "<|en|>",
    "ja": "<|ja|>",
    "yue": "<|yue|>",
    "ko": "<|ko|>",
}
ORIGINAL_ROW_ID_COLUMN = "__cosyvoice_original_row_id"

_T = TypeVar("_T")


def normalize_header(header_name: str) -> str:
    return " ".join(header_name.strip().lower().replace("_", " ").split())


def resolve_columns(fieldnames: Sequence[str]) -> Dict[str, str]:
    column_mapping: Dict[str, str] = {}
    for column_name in fieldnames:
        normalized_name = normalize_header(column_name)
        if normalized_name == "id":
            column_mapping["id"] = column_name
        if normalized_name == "cosyvoice original row id":
            column_mapping["row_id"] = column_name
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


def load_rows(input_tsv_path: Path) -> List["TsvInputRow"]:
    from batch_types import TsvInputRow

    rows: List["TsvInputRow"] = []
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
            row_id = str(row_index)
            if "row_id" in column_mapping:
                row_id = (row.get(column_mapping["row_id"], "") or row_id).strip()
            input_id = None
            if "id" in column_mapping:
                input_id = (row.get(column_mapping["id"], "") or "").strip()
            rows.append(
                TsvInputRow(
                    output_index=output_index,
                    row_id=row_id,
                    text=text,
                    ref_audio_path=ref_audio_path,
                    input_id=input_id,
                )
            )
            output_index += 1
    return rows


def chunked(items: Sequence[_T], chunk_size: int) -> Iterator[Sequence[_T]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]


def count_tsv_rows(tsv_path: Path) -> int:
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return 0
    with tsv_path.open("r", encoding="utf-8", newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        if not reader.fieldnames:
            return 0
        return sum(1 for row in reader if None not in row)


def write_metadata(
    output_tsv_path: Path,
    metadata_rows: Iterable[Dict[str, str]],
    include_input_id: bool = False,
) -> None:
    fieldnames = metadata_fieldnames(include_input_id)
    with output_tsv_path.open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=fieldnames,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(metadata_rows)


def metadata_fieldnames(include_input_id: bool = False) -> List[str]:
    fieldnames = ["speechpath", "text"]
    if include_input_id:
        fieldnames.append("id")
    return fieldnames


def append_metadata(
    output_tsv_path: Path,
    metadata_rows: Iterable[Dict[str, str]],
    include_input_id: bool = False,
) -> None:
    rows = list(metadata_rows)
    if not rows:
        return
    with output_tsv_path.open("a", encoding="utf-8", newline="") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=metadata_fieldnames(include_input_id),
            delimiter="\t",
        )
        writer.writerows(rows)


def write_failures(
    failure_tsv_path: Path,
    failure_rows: Iterable[Dict[str, str]],
    include_input_id: bool = False,
) -> None:
    fieldnames = failure_fieldnames(include_input_id)
    with failure_tsv_path.open("w", encoding="utf-8", newline="") as failure_file:
        writer = csv.DictWriter(
            failure_file,
            fieldnames=fieldnames,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(failure_rows)


def failure_fieldnames(include_input_id: bool = False) -> List[str]:
    fieldnames = ["row_id"]
    if include_input_id:
        fieldnames.append("id")
    fieldnames.extend(["text", "ref_audio", "error"])
    return fieldnames


def append_failures(
    failure_tsv_path: Path,
    failure_rows: Iterable[Dict[str, str]],
    include_input_id: bool = False,
) -> None:
    rows = list(failure_rows)
    if not rows:
        return
    with failure_tsv_path.open("a", encoding="utf-8", newline="") as failure_file:
        writer = csv.DictWriter(
            failure_file,
            fieldnames=failure_fieldnames(include_input_id),
            delimiter="\t",
        )
        writer.writerows(rows)
