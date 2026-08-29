#!/usr/bin/env python3
"""Convert every candidate in generated.json to TTS training metadata.

Unlike process_asr_judged_tts.py, this tool performs no ASR-based filtering.
Every candidate audio path belonging to a selected output item is resampled and
written to the output dataset.

Examples:
  python3 process_generated_tts.py \
    --input /path/to/output/generated.json \
    --output-dir /path/to/prepared_tts

  # Keep output indexes 0 through 16 (inclusive) from every group.
  python3 process_generated_tts.py \
    --input /path/to/output/generated.json \
    --output-dir /path/to/prepared_wuw \
    --group-output-index-range 0:16

Relative candidate_audio_path values are resolved from the directory containing
generated.json. Use --audio-parent-dir only when they have a different base.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class GeneratedCandidate:
    group_index: int
    group_id: str
    output_index: int
    candidate_index: int
    text: str
    source_audio: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resample every candidate from generated.json and write "
            "CosyVoice-style metadata.jsonl without ASR filtering."
        )
    )
    parser.add_argument(
        "--input",
        "--generated-json",
        dest="input",
        required=True,
        help="Input generated.json file.",
    )
    parser.add_argument(
        "--audio-parent-dir",
        "--source-audio-parent-dir",
        dest="audio_parent_dir",
        default=None,
        help=(
            "Only needed when candidate_audio_path entries are relative to a "
            "directory other than the input JSON directory."
        ),
    )
    parser.add_argument(
        "--group-output-index-range",
        "--output-index-range",
        dest="group_output_index_range",
        default=None,
        help=(
            "Inclusive output item index range inside every group. Formats: "
            "START:END, START:, :END, or INDEX. Example: 0:9."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. The script creates wav/ and metadata.jsonl inside it.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Output WAV sample rate in Hz. Default: 16000.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="Optional output channel count, for example 1 for mono. Default: preserve.",
    )
    parser.add_argument(
        "--metadata-name",
        default="metadata.jsonl",
        help="Metadata filename inside output-dir. Default: metadata.jsonl.",
    )
    parser.add_argument(
        "--wav-dir-name",
        default="wav",
        help="WAV subdirectory name inside output-dir. Default: wav.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First numeric WAV id. Default: 0.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=9,
        help="Zero-padding width for WAV filenames. Default: 9.",
    )
    parser.add_argument(
        "--sox",
        default="sox",
        help="sox executable used for WAV resampling. Default: sox.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Warn and skip candidates whose source audio is missing.",
    )
    return parser.parse_args()


def load_json_or_jsonl(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as json_error:
        records = []
        for line_no, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                raise SystemExit(
                    f"{path}:{line_no}: invalid JSON. The file is neither valid "
                    f"JSON nor valid JSONL. First JSON error: {json_error}"
                ) from json_error
        return records


def iter_groups(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, list):
        yield from data
        return
    if isinstance(data, dict):
        for field in ("groups", "inputs", "items", "results", "data"):
            value = data.get(field)
            if isinstance(value, list):
                yield from value
                return
    raise SystemExit(
        "input must be a JSON array, or a JSON object containing one of these "
        "list fields: groups, inputs, items, results, data"
    )


def parse_output_index_range(raw_range: str | None) -> tuple[int, int | None]:
    if raw_range is None or raw_range.strip() == "":
        return (0, None)

    text = raw_range.strip()
    separator = ":" if ":" in text else "-" if "-" in text else None
    if separator is None:
        start_text = text
        end_text = text
    else:
        start_text, end_text = (part.strip() for part in text.split(separator, 1))

    try:
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else None
    except ValueError as exc:
        raise SystemExit(
            "--group-output-index-range must use integer indexes, for example 0:9"
        ) from exc
    if start < 0:
        raise SystemExit("--group-output-index-range start must be non-negative")
    if end is not None:
        if end < 0:
            raise SystemExit("--group-output-index-range end must be non-negative")
        if end < start:
            raise SystemExit(
                "--group-output-index-range end must be greater than or equal to start"
            )
    return (start, end)


def output_index_in_range(
    output_index: int, range_start: int, range_end: int | None
) -> bool:
    return output_index >= range_start and (
        range_end is None or output_index <= range_end
    )


def resolve_audio_path(raw_audio: Any, audio_parent: Path, context: str) -> Path:
    if raw_audio is None or str(raw_audio).strip() == "":
        raise SystemExit(f"{context}: missing candidate audio path")
    path = Path(str(raw_audio)).expanduser()
    if path.is_absolute():
        return path
    return audio_parent / path


def iter_generated_candidates(
    data: Any,
    audio_parent: Path,
    range_start: int,
    range_end: int | None,
) -> Iterable[GeneratedCandidate]:
    for group_index, group in enumerate(iter_groups(data)):
        if not isinstance(group, dict):
            raise SystemExit(f"group #{group_index} is not a JSON object")
        group_id = str(group.get("id", group.get("input_id", group_index)))
        outputs = group.get("output")
        if not isinstance(outputs, list):
            raise SystemExit(
                f"group #{group_index} (id={group_id!r}) is missing list field 'output'"
            )

        for output_index, output in enumerate(outputs):
            if not output_index_in_range(output_index, range_start, range_end):
                continue
            context = (
                f"group #{group_index} (id={group_id!r}), "
                f"output #{output_index}"
            )
            if not isinstance(output, dict):
                raise SystemExit(f"{context} is not a JSON object")
            text = output.get("text")
            if text is None:
                raise SystemExit(f"{context} is missing required field 'text'")
            paths = output.get("candidate_audio_path")
            if not isinstance(paths, list):
                raise SystemExit(
                    f"{context} is missing list field 'candidate_audio_path'"
                )

            for candidate_index, raw_audio in enumerate(paths):
                candidate_context = f"{context}, candidate #{candidate_index}"
                yield GeneratedCandidate(
                    group_index=group_index,
                    group_id=group_id,
                    output_index=output_index,
                    candidate_index=candidate_index,
                    text=str(text),
                    source_audio=resolve_audio_path(
                        raw_audio, audio_parent, candidate_context
                    ),
                )


def run_sox(
    sox_bin: str,
    source: Path,
    target: Path,
    sample_rate: int,
    channels: int | None,
) -> None:
    command = [sox_bin, str(source)]
    if channels is not None:
        command.extend(["-c", str(channels)])
    command.extend(["-r", str(sample_rate), str(target)])
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        message = f"sox failed for {source} -> {target}"
        if stderr:
            message += f": {stderr}"
        raise RuntimeError(message) from exc


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be greater than 0")
    if args.channels is not None and args.channels <= 0:
        raise SystemExit("--channels must be greater than 0 when provided")
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if args.digits <= 0:
        raise SystemExit("--digits must be greater than 0")
    for option, value in (
        ("--metadata-name", args.metadata_name),
        ("--wav-dir-name", args.wav_dir_name),
    ):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"{option} must stay inside --output-dir")


def main() -> int:
    args = parse_args()
    validate_args(args)
    range_start, range_end = parse_output_index_range(
        args.group_output_index_range
    )

    sox_path = shutil.which(args.sox)
    if sox_path is None:
        raise SystemExit(
            f"cannot find sox executable {args.sox!r}; install sox or pass --sox"
        )

    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        raise SystemExit(f"input generated JSON not found: {input_path}")

    audio_parent = (
        Path(args.audio_parent_dir).expanduser()
        if args.audio_parent_dir
        else input_path.parent
    )
    if not audio_parent.is_dir():
        raise SystemExit(f"audio parent dir not found: {audio_parent}")

    output_dir = Path(args.output_dir).expanduser()
    wav_dir = output_dir / args.wav_dir_name
    metadata_path = output_dir / args.metadata_name
    metadata_tmp_path = metadata_path.with_name(metadata_path.name + ".tmp")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    data = load_json_or_jsonl(input_path)
    candidates_seen = 0
    candidates_written = 0
    missing_skipped = 0
    groups_seen: set[int] = set()
    outputs_seen: set[tuple[int, int]] = set()
    next_audio_index = args.start_index

    with metadata_tmp_path.open("w", encoding="utf-8") as metadata_file:
        for candidate in iter_generated_candidates(
            data=data,
            audio_parent=audio_parent,
            range_start=range_start,
            range_end=range_end,
        ):
            candidates_seen += 1
            groups_seen.add(candidate.group_index)
            outputs_seen.add((candidate.group_index, candidate.output_index))
            if not candidate.source_audio.is_file():
                if args.skip_missing:
                    print(
                        "WARNING: skip missing audio: "
                        f"{candidate.source_audio} "
                        f"(group_id={candidate.group_id}, "
                        f"output={candidate.output_index}, "
                        f"candidate={candidate.candidate_index})",
                        file=sys.stderr,
                    )
                    missing_skipped += 1
                    continue
                raise SystemExit(
                    "source audio does not exist: "
                    f"{candidate.source_audio} "
                    f"(group_id={candidate.group_id}, "
                    f"output={candidate.output_index}, "
                    f"candidate={candidate.candidate_index})"
                )

            wav_name = f"{next_audio_index:0{args.digits}d}.wav"
            relative_audio = Path(args.wav_dir_name) / wav_name
            target_audio = output_dir / relative_audio
            run_sox(
                sox_bin=sox_path,
                source=candidate.source_audio,
                target=target_audio,
                sample_rate=args.sample_rate,
                channels=args.channels,
            )
            record = {
                "audiofile_path": relative_audio.as_posix(),
                "text": candidate.text,
            }
            metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            candidates_written += 1
            next_audio_index += 1

    metadata_tmp_path.replace(metadata_path)

    print(f"groups_selected={len(groups_seen)}")
    print(f"outputs_selected={len(outputs_seen)}")
    print(f"candidates_seen={candidates_seen}")
    print(f"candidates_written={candidates_written}")
    print(f"missing_skipped={missing_skipped}")
    print(f"wav_dir={wav_dir}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
