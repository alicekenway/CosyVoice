#!/usr/bin/env python3
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio

LANG_TOKEN_MAP = {
    "zh": "<|zh|>",
    "en": "<|en|>",
    "ja": "<|ja|>",
    "yue": "<|yue|>",
    "ko": "<|ko|>",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prepare_import_path() -> None:
    root = _repo_root()
    cosyvoice_root = root / "CosyVoice"
    matcha_root = cosyvoice_root / "third_party" / "Matcha-TTS"
    sys.path.insert(0, str(cosyvoice_root))
    sys.path.insert(0, str(matcha_root))


def _normalize_header(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def _resolve_columns(fieldnames: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for col in fieldnames:
        norm = _normalize_header(col)
        if norm == "text":
            mapping["text"] = col
        if norm in {
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
            mapping["ref_audio"] = col
    return mapping


def _load_rows(input_tsv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with input_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Input TSV has no header: {input_tsv}")
        cols = _resolve_columns(reader.fieldnames)
        if "text" not in cols or "ref_audio" not in cols:
            raise ValueError(
                "Input TSV must contain columns for text and reference audio path. "
                f"Found header: {reader.fieldnames}"
            )
        for idx, row in enumerate(reader, start=1):
            text = (row.get(cols["text"], "") or "").strip()
            ref_audio = (row.get(cols["ref_audio"], "") or "").strip()
            if not text or not ref_audio:
                continue
            rows.append({"text": text, "ref_audio": ref_audio, "row_id": str(idx)})
    return rows


def _concat_segments(segments: List[torch.Tensor]) -> torch.Tensor:
    if not segments:
        raise RuntimeError("No generated segments returned by model")
    if len(segments) == 1:
        return segments[0]
    return torch.cat(segments, dim=1)


def _speech_duration_sec(speech: torch.Tensor, sample_rate: int) -> float:
    if speech.ndim < 2:
        raise ValueError(f"Expected speech tensor with shape [channels, samples], got {tuple(speech.shape)}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    return float(speech.shape[1]) / float(sample_rate)


def _safe_rtf(runtime_sec: float, audio_sec: float) -> float:
    if audio_sec <= 0.0:
        return float("inf")
    return runtime_sec / audio_sec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch TTS generation with CosyVoice from TSV input."
    )
    parser.add_argument("--model_path", required=True, help="Local model directory path")
    parser.add_argument(
        "--input_tsv",
        required=True,
        help="TSV file with 2 columns: text and reference audio path",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory (creates wav/ and metadata TSV)",
    )
    parser.add_argument(
        "--output_tsv_name",
        default="generated.tsv",
        help="Name of output TSV inside output_dir (default: generated.tsv)",
    )
    parser.add_argument(
        "--text_frontend",
        action="store_true",
        help="Enable text frontend normalization (default: disabled)",
    )
    parser.add_argument(
        "--lang",
        choices=list(LANG_TOKEN_MAP.keys()),
        default=None,
        help="Language tag to prefix text (recommended): zh/en/ja/yue/ko",
    )
    args = parser.parse_args()

    _prepare_import_path()
    from cosyvoice.cli.cosyvoice import AutoModel  # pylint: disable=import-outside-toplevel

    input_tsv = Path(args.input_tsv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    wav_dir = output_dir / "wav"
    output_tsv = output_dir / args.output_tsv_name

    rows = _load_rows(input_tsv)
    if not rows:
        raise ValueError(f"No valid rows found in input TSV: {input_tsv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    cosyvoice = AutoModel(model_dir=str(Path(args.model_path).expanduser().resolve()))

    metadata_rows: List[Dict[str, str]] = []
    total_runtime_sec = 0.0
    total_audio_sec = 0.0
    lang_token = LANG_TOKEN_MAP[args.lang] if args.lang else ""
    for i, row in enumerate(rows):
        row_start_time = time.perf_counter()
        text = row["text"]
        if lang_token and not text.startswith(lang_token):
            text = f"{lang_token}{text}"
        ref_audio = str(Path(row["ref_audio"]).expanduser().resolve())
        file_name = f"utt_{i:06d}.wav"
        rel_speech_path = f"wav/{file_name}"
        abs_speech_path = wav_dir / file_name

        segments: List[torch.Tensor] = []
        for out in cosyvoice.inference_cross_lingual(
            text,
            ref_audio,
            stream=False,
            text_frontend=args.text_frontend,
        ):
            segments.append(out["tts_speech"].cpu())
        speech = _concat_segments(segments)
        torchaudio.save(str(abs_speech_path), speech, cosyvoice.sample_rate)
        row_runtime_sec = time.perf_counter() - row_start_time
        row_audio_sec = _speech_duration_sec(speech, cosyvoice.sample_rate)
        row_rtf = _safe_rtf(row_runtime_sec, row_audio_sec)
        total_runtime_sec += row_runtime_sec
        total_audio_sec += row_audio_sec
        print(
            f"[row {i + 1}/{len(rows)}] time_sec={row_runtime_sec:.3f}, "
            f"audio_sec={row_audio_sec:.3f}, rtf={row_rtf:.4f}"
        )
        metadata_rows.append({"speechpath": rel_speech_path, "text": text})

    with output_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speechpath", "text"], delimiter="\t")
        writer.writeheader()
        writer.writerows(metadata_rows)

    avg_runtime_sec = total_runtime_sec / len(metadata_rows) if metadata_rows else 0.0
    avg_audio_sec = total_audio_sec / len(metadata_rows) if metadata_rows else 0.0
    avg_rtf = _safe_rtf(total_runtime_sec, total_audio_sec)

    print(f"Generated {len(metadata_rows)} utterances")
    print(
        f"Overall timing: total_time_sec={total_runtime_sec:.3f}, "
        f"total_audio_sec={total_audio_sec:.3f}, overall_rtf={avg_rtf:.4f}, "
        f"avg_time_sec={avg_runtime_sec:.3f}, avg_audio_sec={avg_audio_sec:.3f}"
    )
    print(f"WAV directory: {wav_dir}")
    print(f"Metadata TSV: {output_tsv}")


if __name__ == "__main__":
    main()
