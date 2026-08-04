#!/usr/bin/env python3
"""Prepare TTS audio selected from ASR judgment results.

Example:
  python3 tools/process_asr_judged_tts.py \
    --input asr_results.json \
    --source-input generated.json \
    --output-dir prepared_tts \
    --mode top-n \
    --top-n 1 \
    --threshold 0.1 \
    --group-output-index-range 0:9

Use --audio-parent-dir only when candidate audio paths in the JSON are relative.
Use --source-input when the ASR judged audio was made by concatenating all
candidate audio in each group. In that case, selected ASR group candidates are
mapped back to the independent candidate wavs in the original generated JSON.
Use --group-output-index-range to keep only part of each group, for example
one run for positive cases and another run for negative cases.
Use --no-copy with --source-input to reference the original source wavs in
metadata.jsonl without creating an output wav directory or copying audio.
Use --manual-include-list to force known-good candidates, one per line:
  <id>:<candidate index>
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Candidate:
    result_index: int
    result_id: str
    text: str
    candidate_id: Any
    wer: float
    source_audio: Path
    source_order: int


@dataclass(frozen=True)
class SourceGroupIndex:
    by_id: dict[str, dict[str, Any]]
    by_position: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select TTS candidates by ASR WER, resample the audio, and write "
            "CosyVoice-style metadata.jsonl."
        )
    )
    parser.add_argument("--input", required=True, help="ASR judgment JSON file.")
    parser.add_argument(
        "--source-input",
        "--generated-json",
        dest="source_input",
        default=None,
        help=(
            "Original TTS/ASR input JSON, for example generated.json. When set, "
            "the script writes independent per-item wavs from this file instead "
            "of concatenated ASR candidate audio."
        ),
    )
    parser.add_argument(
        "--audio-parent-dir",
        default=None,
        help=(
            "Only needed when candidate audio paths are relative. If omitted, "
            "absolute paths are used directly and relative paths are resolved "
            "from the input JSON directory."
        ),
    )
    parser.add_argument(
        "--source-audio-parent-dir",
        default=None,
        help=(
            "Only needed with --source-input when candidate_audio_path entries "
            "are relative to a directory other than the source input JSON "
            "directory."
        ),
    )
    parser.add_argument(
        "--group-output-index-range",
        "--output-index-range",
        dest="group_output_index_range",
        default=None,
        help=(
            "Inclusive output item index range inside each source group. "
            "Formats: START:END, START:, :END, or INDEX. Only valid with "
            "--source-input. Example: 0:9."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. The script creates wav/ and metadata.jsonl inside it.",
    )
    parser.add_argument(
        "--no-copy",
        "--no_copy",
        action="store_true",
        help=(
            "Do not create an output WAV directory or copy/resample audio. "
            "Write resolved source-input WAV paths directly to metadata. "
            "Requires --source-input."
        ),
    )
    parser.add_argument(
        "--mode",
        default="top-n",
        help="Selection mode: all, top-n, or strict-top-n. Default: top-n.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="N for top-n and strict-top-n modes. Default: 1.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional maximum WER to keep. Candidates with wer > threshold are dropped.",
    )
    parser.add_argument(
        "--manual-include-list",
        "--wave-list",
        "--wav-list",
        dest="manual_include_list",
        default=None,
        help=(
            "Optional text file of manually accepted candidates to include anyway. "
            "Each non-empty line must be '<id>:<candidate index>'."
        ),
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
        "--seed",
        type=int,
        default=20260702,
        help="Random seed used only by strict-top-n tie sampling. Default: 20260702.",
    )
    parser.add_argument(
        "--sox",
        default="sox",
        help="sox executable used for WAV resampling. Default: sox.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Warn and skip selected candidates whose source audio is missing.",
    )
    return parser.parse_args()


def normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "all": "all",
        "top-n": "top-n",
        "topn": "top-n",
        "strict-top-n": "strict-top-n",
        "strict-topn": "strict-top-n",
        "strict": "strict-top-n",
    }
    if normalized not in aliases:
        valid = ", ".join(["all", "top-n", "strict-top-n"])
        raise SystemExit(f"invalid --mode {mode!r}; expected one of: {valid}")
    return aliases[normalized]


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


def iter_result_items(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, dict):
        results = data.get("results")
        if not isinstance(results, list):
            raise SystemExit("input JSON must contain a list field named 'results'")
        yield from results
        return
    if isinstance(data, list):
        yield from data
        return
    raise SystemExit("input must be a JSON object with 'results' or a JSON array")


def iter_source_groups(data: Any) -> Iterable[dict[str, Any]]:
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
        "source input must be a JSON array, or a JSON object containing one "
        "of these list fields: groups, inputs, items, results, data"
    )


def build_source_group_index(data: Any) -> SourceGroupIndex:
    by_id: dict[str, dict[str, Any]] = {}
    by_position: list[dict[str, Any]] = []
    for group_index, group in enumerate(iter_source_groups(data)):
        if not isinstance(group, dict):
            raise SystemExit(f"source group #{group_index} is not a JSON object")
        by_position.append(group)
        group_id = group.get("id", group.get("input_id"))
        if group_id is not None:
            by_id[str(group_id)] = group
    return SourceGroupIndex(by_id=by_id, by_position=by_position)


def resolve_audio_path(raw_audio: Any, audio_parent: Path | None, input_dir: Path) -> Path:
    if raw_audio is None or str(raw_audio).strip() == "":
        raise ValueError("missing candidate audio path")
    path = Path(str(raw_audio)).expanduser()
    if path.is_absolute():
        return path
    base_dir = audio_parent if audio_parent is not None else input_dir
    return base_dir / path


def read_candidates(
    result: dict[str, Any],
    result_index: int,
    audio_parent: Path | None,
    input_dir: Path,
    require_audio: bool = True,
) -> list[Candidate]:
    text = result.get("text")
    if text is None:
        raise SystemExit(f"result #{result_index} is missing required field 'text'")

    details = result.get("detail")
    if not isinstance(details, list):
        raise SystemExit(f"result #{result_index} is missing list field 'detail'")

    result_id = str(result.get("id", result_index))
    candidates = []
    for source_order, detail in enumerate(details):
        if not isinstance(detail, dict):
            raise SystemExit(
                f"result #{result_index} detail #{source_order} is not a JSON object"
            )
        try:
            wer = float(detail["wer"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"result #{result_index} detail #{source_order} has invalid 'wer'"
            ) from exc
        if not math.isfinite(wer) or wer < 0:
            raise SystemExit(
                f"result #{result_index} detail #{source_order} has invalid WER: {wer}"
            )
        raw_audio = detail.get("audio")
        if raw_audio is None or str(raw_audio).strip() == "":
            if require_audio:
                raise SystemExit(
                    f"result #{result_index} detail #{source_order}: "
                    "missing candidate audio path"
                )
            source_audio = Path("")
        else:
            source_audio = resolve_audio_path(raw_audio, audio_parent, input_dir)
        candidates.append(
            Candidate(
                result_index=result_index,
                result_id=result_id,
                text=str(text),
                candidate_id=detail.get("candidate", source_order),
                wer=wer,
                source_audio=source_audio,
                source_order=source_order,
            )
        )
    return candidates


def load_manual_include_list(path: Path | None) -> set[tuple[str, str]]:
    if path is None:
        return set()
    if not path.is_file():
        raise SystemExit(f"manual include list not found: {path}")

    entries: set[tuple[str, str]] = set()
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise SystemExit(
                f"{path}:{line_no}: invalid manual include entry; "
                "expected '<id>:<candidate index>'"
            )
        result_id, candidate_id = line.split(":", 1)
        result_id = result_id.strip()
        candidate_id = normalize_candidate_index(candidate_id)
        if not result_id or not candidate_id:
            raise SystemExit(
                f"{path}:{line_no}: invalid manual include entry; "
                "id and candidate index must both be non-empty"
            )
        entries.add((result_id, candidate_id))
    return entries


def normalize_candidate_index(value: Any) -> str:
    text = str(value).strip()
    if text.isdigit():
        return str(int(text))
    return text


def candidate_manual_key(candidate: Candidate) -> tuple[str, str]:
    return (candidate.result_id, normalize_candidate_index(candidate.candidate_id))


def candidate_index_as_int(candidate: Candidate) -> int:
    normalized = normalize_candidate_index(candidate.candidate_id)
    if not normalized.isdigit():
        raise SystemExit(
            "source input mapping requires numeric candidate indexes; "
            f"result_id={candidate.result_id}, candidate={candidate.candidate_id!r}"
        )
    return int(normalized)


def merge_selected_candidates(
    auto_selected: list[Candidate], manual_selected: list[Candidate]
) -> list[Candidate]:
    selected_by_key: dict[tuple[str, str], Candidate] = {}
    for candidate in auto_selected + manual_selected:
        selected_by_key.setdefault(candidate_manual_key(candidate), candidate)
    return sorted(selected_by_key.values(), key=candidate_sort_key)


def candidate_sort_key(candidate: Candidate) -> tuple[float, int, str]:
    return (candidate.wer, candidate.source_order, str(candidate.candidate_id))


def select_candidates(
    candidates: list[Candidate], mode: str, top_n: int, rng: random.Random
) -> list[Candidate]:
    if mode == "all":
        return candidates

    ordered = sorted(candidates, key=candidate_sort_key)
    if len(ordered) <= top_n:
        return ordered

    if mode == "top-n":
        boundary_wer = ordered[top_n - 1].wer
        return [candidate for candidate in ordered if candidate.wer <= boundary_wer]

    selected: list[Candidate] = []
    cursor = 0
    while cursor < len(ordered) and len(selected) < top_n:
        group_wer = ordered[cursor].wer
        next_cursor = cursor + 1
        while next_cursor < len(ordered) and ordered[next_cursor].wer == group_wer:
            next_cursor += 1
        group = ordered[cursor:next_cursor]
        slots = top_n - len(selected)
        if len(group) <= slots:
            selected.extend(group)
        else:
            sampled = rng.sample(group, slots)
            selected.extend(sorted(sampled, key=candidate_sort_key))
        cursor = next_cursor
    return selected


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


def validate_args(args: argparse.Namespace, mode: str) -> None:
    if mode != "all" and args.top_n <= 0:
        raise SystemExit("--top-n must be greater than 0 for top-n modes")
    if args.threshold is not None:
        if not math.isfinite(args.threshold) or args.threshold < 0:
            raise SystemExit("--threshold must be a non-negative finite number")
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
    if args.group_output_index_range is not None and not args.source_input:
        raise SystemExit("--group-output-index-range requires --source-input")
    if args.source_audio_parent_dir is not None and not args.source_input:
        raise SystemExit("--source-audio-parent-dir requires --source-input")
    if args.no_copy and not args.source_input:
        raise SystemExit(
            "--no-copy requires --source-input; ASR --input audio must not be used "
            "as source audio in no-copy mode"
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
        parts = text.split(separator, 1)
        start_text = parts[0].strip()
        end_text = parts[1].strip()

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


def find_source_group(
    source_groups: SourceGroupIndex,
    candidate: Candidate,
) -> dict[str, Any]:
    group = source_groups.by_id.get(candidate.result_id)
    if group is not None:
        return group
    if candidate.result_index < len(source_groups.by_position):
        return source_groups.by_position[candidate.result_index]
    raise SystemExit(
        "cannot find source group for ASR result; "
        f"result_index={candidate.result_index}, result_id={candidate.result_id}"
    )


def get_source_outputs(group: dict[str, Any], candidate: Candidate) -> list[Any]:
    outputs = group.get("output")
    if not isinstance(outputs, list):
        raise SystemExit(
            "source group is missing list field 'output'; "
            f"result_index={candidate.result_index}, result_id={candidate.result_id}"
        )
    return outputs


def expand_source_candidate_audio(
    candidate: Candidate,
    source_groups: SourceGroupIndex,
    source_audio_parent: Path | None,
    source_input_dir: Path,
    output_range_start: int,
    output_range_end: int | None,
) -> list[Candidate]:
    group = find_source_group(source_groups, candidate)
    outputs = get_source_outputs(group, candidate)
    candidate_index = candidate_index_as_int(candidate)
    expanded: list[Candidate] = []

    for output_index, output in enumerate(outputs):
        if not output_index_in_range(output_index, output_range_start, output_range_end):
            continue
        if not isinstance(output, dict):
            raise SystemExit(
                "source group output item is not a JSON object; "
                f"result_id={candidate.result_id}, output_index={output_index}"
            )

        paths = output.get("candidate_audio_path")
        if not isinstance(paths, list):
            raise SystemExit(
                "source group output item is missing list field "
                "'candidate_audio_path'; "
                f"result_id={candidate.result_id}, output_index={output_index}"
            )
        if candidate_index >= len(paths):
            raise SystemExit(
                "source group output item has no audio for selected candidate; "
                f"result_id={candidate.result_id}, output_index={output_index}, "
                f"candidate={candidate_index}, candidate_count={len(paths)}"
            )

        try:
            source_audio = resolve_audio_path(
                paths[candidate_index], source_audio_parent, source_input_dir
            )
        except ValueError as exc:
            raise SystemExit(
                "source group output item has invalid audio path; "
                f"result_id={candidate.result_id}, output_index={output_index}: {exc}"
            ) from exc

        text = output.get("text")
        if text is None:
            raise SystemExit(
                "source group output item is missing required field 'text'; "
                f"result_id={candidate.result_id}, output_index={output_index}"
            )

        expanded.append(
            Candidate(
                result_index=candidate.result_index,
                result_id=candidate.result_id,
                text=str(text),
                candidate_id=candidate.candidate_id,
                wer=candidate.wer,
                source_audio=source_audio,
                source_order=output_index,
            )
        )
    return expanded


def warn_no_candidate_passed_threshold(
    result: dict[str, Any], result_index: int, threshold: float
) -> None:
    result_id = result.get("id", result_index)
    text = result.get("text", "")
    print(
        "WARNING: no candidate passed threshold; "
        f"id={json.dumps(result_id, ensure_ascii=False)}; "
        f"text={json.dumps(text, ensure_ascii=False)}; "
        f"threshold={threshold}",
        file=sys.stderr,
    )


def main() -> int:
    args = parse_args()
    mode = normalize_mode(args.mode)
    validate_args(args, mode)
    output_range_start, output_range_end = parse_output_index_range(
        args.group_output_index_range
    )

    sox_path: str | None = None
    if not args.no_copy:
        sox_path = shutil.which(args.sox)
        if sox_path is None:
            raise SystemExit(
                f"cannot find sox executable {args.sox!r}; install sox or pass --sox"
            )

    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        raise SystemExit(f"input JSON not found: {input_path}")

    audio_parent = (
        Path(args.audio_parent_dir).expanduser() if args.audio_parent_dir else None
    )
    if audio_parent is not None and not audio_parent.is_dir():
        raise SystemExit(f"audio parent dir not found: {audio_parent}")

    source_input_path = (
        Path(args.source_input).expanduser() if args.source_input else None
    )
    source_groups: SourceGroupIndex | None = None
    source_input_dir: Path | None = None
    if source_input_path is not None:
        if not source_input_path.is_file():
            raise SystemExit(f"source input JSON not found: {source_input_path}")
        source_data = load_json_or_jsonl(source_input_path)
        source_groups = build_source_group_index(source_data)
        source_input_dir = source_input_path.parent

    source_audio_parent = (
        Path(args.source_audio_parent_dir).expanduser()
        if args.source_audio_parent_dir
        else None
    )
    if source_audio_parent is not None and not source_audio_parent.is_dir():
        raise SystemExit(f"source audio parent dir not found: {source_audio_parent}")

    manual_include_path = (
        Path(args.manual_include_list).expanduser()
        if args.manual_include_list
        else None
    )
    manual_include_keys = load_manual_include_list(manual_include_path)
    manual_include_seen: set[tuple[str, str]] = set()

    output_dir = Path(args.output_dir).expanduser()
    wav_dir = output_dir / args.wav_dir_name
    metadata_path = output_dir / args.metadata_name
    metadata_tmp_path = metadata_path.with_name(metadata_path.name + ".tmp")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_copy:
        wav_dir.mkdir(parents=True, exist_ok=True)

    data = load_json_or_jsonl(input_path)
    rng = random.Random(args.seed)
    input_dir = input_path.parent

    results_seen = 0
    candidates_seen = 0
    candidates_after_threshold = 0
    candidates_selected = 0
    threshold_dropped = 0
    missing_skipped = 0
    groups_without_selection = 0
    manual_selected = 0
    asr_candidates_selected = 0
    source_outputs_selected = 0

    next_audio_index = args.start_index
    with metadata_tmp_path.open("w", encoding="utf-8") as metadata_file:
        for result_index, result in enumerate(iter_result_items(data)):
            if not isinstance(result, dict):
                raise SystemExit(f"result #{result_index} is not a JSON object")
            results_seen += 1
            details = result.get("detail") or []
            candidates_seen += len(details)

            candidates = read_candidates(
                result=result,
                result_index=result_index,
                audio_parent=audio_parent,
                input_dir=input_dir,
                require_audio=source_groups is None,
            )
            candidates_passing_threshold = [
                candidate
                for candidate in candidates
                if args.threshold is None or candidate.wer <= args.threshold
            ]
            dropped = len(candidates) - len(candidates_passing_threshold)
            threshold_dropped += dropped
            candidates_after_threshold += len(candidates_passing_threshold)

            auto_selected = select_candidates(
                candidates_passing_threshold, mode, args.top_n, rng
            )
            forced_selected = [
                candidate
                for candidate in candidates
                if candidate_manual_key(candidate) in manual_include_keys
            ]
            manual_include_seen.update(
                candidate_manual_key(candidate) for candidate in forced_selected
            )
            selected = merge_selected_candidates(auto_selected, forced_selected)
            manual_selected += sum(
                1
                for candidate in selected
                if candidate_manual_key(candidate) in manual_include_keys
            )
            if not selected:
                if args.threshold is not None and dropped == len(details) and details:
                    warn_no_candidate_passed_threshold(
                        result=result,
                        result_index=result_index,
                        threshold=args.threshold,
                    )
                groups_without_selection += 1
                continue
            asr_candidates_selected += len(selected)

            selected_audio: list[Candidate] = []
            if source_groups is None:
                selected_audio = selected
            else:
                assert source_input_dir is not None
                for candidate in selected:
                    expanded = expand_source_candidate_audio(
                        candidate=candidate,
                        source_groups=source_groups,
                        source_audio_parent=source_audio_parent,
                        source_input_dir=source_input_dir,
                        output_range_start=output_range_start,
                        output_range_end=output_range_end,
                    )
                    source_outputs_selected += len(expanded)
                    selected_audio.extend(expanded)
                if not selected_audio:
                    groups_without_selection += 1
                    continue

            for candidate in selected_audio:
                if not candidate.source_audio.is_file():
                    if args.skip_missing:
                        print(f"skip missing audio: {candidate.source_audio}")
                        missing_skipped += 1
                        continue
                    raise SystemExit(
                        "selected source audio does not exist: "
                        f"{candidate.source_audio} "
                        f"(result_id={candidate.result_id}, "
                        f"candidate={candidate.candidate_id})"
                    )

                if args.no_copy:
                    metadata_audio_path = candidate.source_audio.resolve().as_posix()
                else:
                    assert sox_path is not None
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
                    metadata_audio_path = relative_audio.as_posix()
                    next_audio_index += 1

                record = {
                    "audiofile_path": metadata_audio_path,
                    "text": candidate.text,
                }
                metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                candidates_selected += 1

    metadata_tmp_path.replace(metadata_path)

    print(f"results_seen={results_seen}")
    print(f"candidates_seen={candidates_seen}")
    print(f"candidates_after_threshold={candidates_after_threshold}")
    print(f"candidates_selected={candidates_selected}")
    print(f"asr_candidates_selected={asr_candidates_selected}")
    print(f"source_outputs_selected={source_outputs_selected}")
    print(f"threshold_dropped={threshold_dropped}")
    print(f"missing_skipped={missing_skipped}")
    print(f"groups_without_selection={groups_without_selection}")
    print(f"manual_list_entries={len(manual_include_keys)}")
    print(f"manual_selected={manual_selected}")
    missing_manual_keys = sorted(manual_include_keys - manual_include_seen)
    print(f"manual_missing={len(missing_manual_keys)}")
    for result_id, candidate_id in missing_manual_keys:
        print(
            "WARNING: manual include entry was not found in ASR results; "
            f"id={json.dumps(result_id, ensure_ascii=False)}; "
            f"candidate={json.dumps(candidate_id, ensure_ascii=False)}",
            file=sys.stderr,
        )
    if args.no_copy:
        print("wav_dir=not created (--no-copy)")
    else:
        print(f"wav_dir={wav_dir}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
