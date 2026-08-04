#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple

"""
Build CosyVoice v4 JSON input from target texts and an ASR JSONL dataset.
Usually, we use it to generate a block of text with different speakers, for wuw.

Example:

python3 prepare_cosyvoice_input_json.py \
  --text-file /path/to/text.txt \
  --asr-jsonl /path/to/asr_dataset.jsonl \
  --replication-index 300 \
  --candidate-number 3 \
  --output-json /mnt/users/jinyang_wang/TTS_cosyvoice/generation/ENX/batch_2/input_tsv/control_expanded_test.json
"""


JsonRecord = Dict[str, Any]
SpeakerAudioMap = Dict[str, List[str]]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be 0 or greater")
    return parsed


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


def load_speaker_audio(
    jsonl_path: Path,
    speaker_key: str,
    audio_key: str,
) -> Tuple[SpeakerAudioMap, int, int]:
    audio_by_speaker: DefaultDict[str, List[str]] = defaultdict(list)
    seen_by_speaker: DefaultDict[str, set[str]] = defaultdict(set)
    total_records = 0
    skipped_records = 0

    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, raw_line in enumerate(jsonl_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {jsonl_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {jsonl_path}:{line_number}")

            speaker = str(record.get(speaker_key, "")).strip()
            audio_path = str(record.get(audio_key, "")).strip()
            if not speaker or not audio_path:
                skipped_records += 1
                continue
            if audio_path in seen_by_speaker[speaker]:
                skipped_records += 1
                continue
            seen_by_speaker[speaker].add(audio_path)
            audio_by_speaker[speaker].append(audio_path)

    if not audio_by_speaker:
        raise ValueError(
            f"No usable '{speaker_key}' and '{audio_key}' pairs found in {jsonl_path}"
        )
    return dict(audio_by_speaker), total_records, skipped_records


def eligible_speakers(
    audio_by_speaker: SpeakerAudioMap,
    candidate_number: int,
) -> List[str]:
    speakers = [
        speaker
        for speaker, audio_paths in audio_by_speaker.items()
        if len(audio_paths) >= candidate_number
    ]
    if not speakers:
        max_refs = max(len(audio_paths) for audio_paths in audio_by_speaker.values())
        raise ValueError(
            f"No speaker has at least {candidate_number} unique audio paths. "
            f"The largest speaker has {max_refs}."
        )
    return speakers


def choose_speaker(
    speakers: Sequence[str],
    record_index: int,
    rng: random.Random,
    selection: str,
) -> str:
    if selection == "random":
        return rng.choice(list(speakers))
    return speakers[record_index % len(speakers)]


def build_records(
    texts: Sequence[str],
    audio_by_speaker: SpeakerAudioMap,
    speakers: Sequence[str],
    replication_index: int,
    candidate_number: int,
    id_prefix: str,
    rng: random.Random,
    selection: str,
) -> List[JsonRecord]:
    records: List[JsonRecord] = []

    for record_index in range(replication_index):
        speaker = choose_speaker(speakers, record_index, rng, selection)
        reference_audio_paths = rng.sample(
            audio_by_speaker[speaker],
            candidate_number,
        )
        records.append(
            {
                "id": f"{id_prefix}_{record_index:06d}",
                "text": list(texts),
                "reference_audio_path": reference_audio_paths,
            }
        )

    return records


def write_json(output_json: Path, records: Sequence[JsonRecord]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CosyVoice v4 JSON input. Each output record contains the "
            "whole target text file and N reference audios sampled from one speaker."
        )
    )
    parser.add_argument(
        "--text-file",
        required=True,
        type=Path,
        help=(
            "Text file with one target sentence per non-empty line. All non-empty "
            "lines are placed into every output group."
        ),
    )
    parser.add_argument(
        "--asr-jsonl",
        "--audio-jsonl",
        dest="asr_jsonl",
        required=True,
        type=Path,
        help=(
            "ASR JSONL dataset. Each line should contain speaker and "
            "audio_filepath fields."
        ),
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Output CosyVoice v4 JSON path.",
    )
    parser.add_argument(
        "--replication-index",
        "--replications",
        "--speaker-groups",
        type=positive_int,
        default=1,
        help=(
            "Number of output speaker groups to create. Each group contains the "
            "full text file (default: 1)."
        ),
    )
    parser.add_argument(
        "--candidate-number",
        "--candidates-per-text",
        type=positive_int,
        default=3,
        help=(
            "Number of same-speaker reference audios for each output group "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--speaker-key",
        default="speaker",
        help="JSONL key containing the speaker id (default: speaker).",
    )
    parser.add_argument(
        "--audio-key",
        default="audio_filepath",
        help="JSONL key containing the audio path (default: audio_filepath).",
    )
    parser.add_argument(
        "--id-prefix",
        default="group",
        help="Output id prefix (default: group).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for speaker shuffling and audio sampling (default: 0).",
    )
    parser.add_argument(
        "--no-shuffle-speakers",
        action="store_true",
        help=(
            "Keep eligible speakers in their first-appearance order from the ASR "
            "JSONL. This mode is required when using --start-from-speaker."
        ),
    )
    parser.add_argument(
        "--start-from-speaker",
        type=nonnegative_int,
        default=None,
        help=(
            "In --no-shuffle-speakers mode, skip the first N eligible speakers "
            "before speaker selection (default: 0)."
        ),
    )
    parser.add_argument(
        "--speaker-selection",
        choices=("round-robin", "random"),
        default="round-robin",
        help=(
            "How to choose speakers across output groups. By default, eligible "
            "speakers are first shuffled with --seed (default: round-robin)."
        ),
    )
    args = parser.parse_args()
    if args.start_from_speaker is not None and not args.no_shuffle_speakers:
        parser.error(
            "--start-from-speaker requires --no-shuffle-speakers because an "
            "offset is not stable when speakers are shuffled"
        )
    if args.start_from_speaker is None:
        args.start_from_speaker = 0
    return args


def main() -> None:
    args = parse_args()
    text_path = args.text_file.expanduser()
    jsonl_path = args.asr_jsonl.expanduser()
    output_json = args.output_json.expanduser()

    texts = load_texts(text_path)
    audio_by_speaker, total_records, skipped_records = load_speaker_audio(
        jsonl_path,
        args.speaker_key,
        args.audio_key,
    )

    rng = random.Random(args.seed)
    speakers = eligible_speakers(audio_by_speaker, args.candidate_number)
    eligible_speaker_count = len(speakers)
    if args.no_shuffle_speakers:
        if args.start_from_speaker >= eligible_speaker_count:
            raise ValueError(
                f"--start-from-speaker {args.start_from_speaker} is outside the "
                f"range of {eligible_speaker_count} eligible speakers"
            )
        speakers = speakers[args.start_from_speaker :]
    else:
        rng.shuffle(speakers)

    records = build_records(
        texts=texts,
        audio_by_speaker=audio_by_speaker,
        speakers=speakers,
        replication_index=args.replication_index,
        candidate_number=args.candidate_number,
        id_prefix=args.id_prefix,
        rng=rng,
        selection=args.speaker_selection,
    )
    write_json(output_json, records)

    print(f"Text lines: {len(texts)}")
    print(f"Output groups requested: {args.replication_index}")
    print(f"Output groups written: {len(records)}")
    print(f"Text lines per group: {len(texts)}")
    print(f"Candidates per group: {args.candidate_number}")
    print(f"ASR JSONL records read: {total_records}")
    print(f"ASR JSONL records skipped: {skipped_records}")
    print(f"Speakers found: {len(audio_by_speaker)}")
    print(f"Eligible speakers found: {eligible_speaker_count}")
    print(f"Speaker shuffling enabled: {not args.no_shuffle_speakers}")
    print(f"Speakers skipped from start: {args.start_from_speaker}")
    print(f"Eligible speakers available for selection: {len(speakers)}")
    print(f"Output JSON: {output_json}")


if __name__ == "__main__":
    main()
