#!/usr/bin/env python3
import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torchaudio

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from batch_types import PreparedRow, SynthesisResult, TsvInputRow
from frontend_batch import CrossLingualBatchPreparer
from io_utils import LANG_TOKEN_MAP, chunked, load_rows, write_failures, write_metadata
from staged_inference import StagedBatchInferenceRunner

def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        # New layout: inference_scripts/ and cosyvoice/ are siblings.
        if (parent / "inference_scripts").is_dir() and (parent / "cosyvoice").is_dir():
            return parent
    raise RuntimeError("Unable to locate project root containing inference_scripts/ and cosyvoice/")


def prepare_import_path() -> None:
    root = repo_root()
    cosyvoice_root = root
    matcha_root = cosyvoice_root / "third_party" / "Matcha-TTS"
    if str(cosyvoice_root) not in sys.path:
        sys.path.insert(0, str(cosyvoice_root))
    if str(matcha_root) not in sys.path:
        sys.path.insert(0, str(matcha_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Staged batch TTS generation with CosyVoice from TSV input."
    )
    parser.add_argument("--model_path", required=True, help="Local model directory path")
    parser.add_argument(
        "--input_tsv",
        required=True,
        help="TSV file with columns for text and reference audio path",
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
        "--failed_tsv_name",
        default="failed.tsv",
        help="Name of failed-row TSV inside output_dir (default: failed.tsv)",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of TSV rows processed together",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Micro-batch size for LLM stage (default: batch_size)",
    )
    parser.add_argument(
        "--flow_batch_size",
        type=int,
        default=None,
        help="Micro-batch size for flow+vocoder stage (default: batch_size)",
    )
    parser.add_argument(
        "--min_token_text_ratio",
        type=float,
        default=2.0,
        help="Minimum generated speech-token/text-token ratio",
    )
    parser.add_argument(
        "--max_token_text_ratio",
        type=float,
        default=20.0,
        help="Maximum generated speech-token/text-token ratio",
    )
    parser.add_argument(
        "--flow_n_timesteps",
        type=int,
        default=10,
        help="Flow diffusion steps (lower is faster but lower quality)",
    )
    parser.add_argument(
        "--on_error",
        choices=["raise", "skip"],
        default="skip",
        help="Error policy for row-level failures (default: skip)",
    )
    parser.add_argument(
        "--save_workers",
        type=int,
        default=4,
        help="Number of threads used to save wav files (default: 4)",
    )
    return parser.parse_args()


def save_result(
    result: SynthesisResult,
    wav_dir: Path,
    sample_rate: int,
) -> Dict[str, str]:
    file_name = f"utt_{result.row.output_index:06d}.wav"
    abs_speech_path = wav_dir / file_name
    rel_speech_path = f"wav/{file_name}"
    torchaudio.save(str(abs_speech_path), result.speech, sample_rate)
    return {"speechpath": rel_speech_path, "text": result.text_for_metadata}


def prepare_batch_rows(
    row_batch: List[TsvInputRow],
    preparer: CrossLingualBatchPreparer,
    on_error: str,
) -> tuple[List[PreparedRow], List[Dict[str, str]]]:
    prepared_rows: List[PreparedRow] = []
    failure_rows: List[Dict[str, str]] = []
    for row in row_batch:
        try:
            prepared_row = preparer.prepare_rows([row])[0]
        except Exception as exc:  # pylint: disable=broad-except
            if on_error == "raise":
                raise RuntimeError(f"Frontend preparation failed for row_id={row.row_id}: {exc}") from exc
            failure_rows.append(
                {
                    "row_id": row.row_id,
                    "text": row.text,
                    "ref_audio": row.ref_audio_path,
                    "error": f"Frontend preparation failed: {exc}",
                }
            )
            continue
        prepared_rows.append(prepared_row)
    return prepared_rows, failure_rows


def speech_duration_sec(speech, sample_rate: int) -> float:
    if speech.ndim < 2:
        raise ValueError(f"Expected speech tensor with shape [channels, samples], got {tuple(speech.shape)}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    return float(speech.shape[1]) / float(sample_rate)


def safe_rtf(runtime_sec: float, audio_sec: float) -> float:
    if audio_sec <= 0.0:
        return float("inf")
    return runtime_sec / audio_sec


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    llm_batch_size = args.llm_batch_size or args.batch_size
    flow_batch_size = args.flow_batch_size or args.batch_size
    if llm_batch_size <= 0 or flow_batch_size <= 0:
        raise ValueError("--llm_batch_size and --flow_batch_size must be positive")
    if args.min_token_text_ratio <= 0 or args.max_token_text_ratio <= 0:
        raise ValueError("--min_token_text_ratio and --max_token_text_ratio must be positive")
    if args.max_token_text_ratio < args.min_token_text_ratio:
        raise ValueError("--max_token_text_ratio must be >= --min_token_text_ratio")
    if args.flow_n_timesteps <= 0:
        raise ValueError("--flow_n_timesteps must be positive")
    if args.save_workers <= 0:
        raise ValueError("--save_workers must be positive")

    prepare_import_path()
    from cosyvoice.cli.cosyvoice import AutoModel  # pylint: disable=import-outside-toplevel
    import hyperpyyaml  # pylint: disable=import-outside-toplevel
    import ruamel.yaml  # pylint: disable=import-outside-toplevel
    if not hasattr(ruamel.yaml.Loader, "max_depth"):
        ruamel.yaml.Loader.max_depth = None
    if hasattr(ruamel.yaml, "SafeLoader") and not hasattr(ruamel.yaml.SafeLoader, "max_depth"):
        ruamel.yaml.SafeLoader.max_depth = None
    if hasattr(ruamel.yaml, "FullLoader") and not hasattr(ruamel.yaml.FullLoader, "max_depth"):
        ruamel.yaml.FullLoader.max_depth = None
    if hasattr(ruamel.yaml, "UnsafeLoader") and not hasattr(ruamel.yaml.UnsafeLoader, "max_depth"):
        ruamel.yaml.UnsafeLoader.max_depth = None

    input_tsv_path = Path(args.input_tsv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    wav_dir = output_dir / "wav"
    output_tsv_path = output_dir / args.output_tsv_name
    failed_tsv_path = output_dir / args.failed_tsv_name

    rows = load_rows(input_tsv_path)
    if not rows:
        raise ValueError(f"No valid rows found in input TSV: {input_tsv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    try:
        cosyvoice = AutoModel(model_dir=str(Path(args.model_path).expanduser().resolve()))
    except Exception as exc:  # pylint: disable=broad-except
        raise
    lang_token = LANG_TOKEN_MAP[args.lang] if args.lang else ""
    preparer = CrossLingualBatchPreparer(
        frontend=cosyvoice.frontend,
        sample_rate=cosyvoice.sample_rate,
        text_frontend=args.text_frontend,
        lang_token=lang_token,
    )
    runner = StagedBatchInferenceRunner(
        cosyvoice=cosyvoice,
        llm_batch_size=llm_batch_size,
        flow_batch_size=flow_batch_size,
        on_error=args.on_error,
        min_token_text_ratio=args.min_token_text_ratio,
        max_token_text_ratio=args.max_token_text_ratio,
        flow_n_timesteps=args.flow_n_timesteps,
    )

    metadata_rows: List[Dict[str, str]] = []
    save_futures: List[Tuple[int, Future[Dict[str, str]]]] = []
    failure_rows: List[Dict[str, str]] = []
    total_runtime_sec = 0.0
    save_wait_sec = 0.0
    total_audio_sec = 0.0
    successful_batch_count = 0
    save_executor = ThreadPoolExecutor(max_workers=args.save_workers)

    try:
        for batch_index, row_batch in enumerate(chunked(rows, args.batch_size), start=1):
            batch_start_time = time.perf_counter()
            prepared_rows, frontend_failures = prepare_batch_rows(
                row_batch=row_batch,
                preparer=preparer,
                on_error=args.on_error,
            )
            failure_rows.extend(frontend_failures)
            if not prepared_rows:
                continue

            batch_success_count = 0
            batch_audio_sec = 0.0
            for result in runner.run_batch(prepared_rows):
                if result.is_success:
                    item_audio_sec = speech_duration_sec(result.speech, cosyvoice.sample_rate)
                    save_futures.append(
                        (
                            result.row.output_index,
                            save_executor.submit(
                                save_result,
                                result=result,
                                wav_dir=wav_dir,
                                sample_rate=cosyvoice.sample_rate,
                            ),
                        )
                    )
                    batch_success_count += 1
                    batch_audio_sec += item_audio_sec
                else:
                    if args.on_error == "raise":
                        raise RuntimeError(
                            f"Inference failed for row_id={result.row.row_id}: {result.error_message}"
                        )
                    failure_rows.append(
                        {
                            "row_id": result.row.row_id,
                            "text": result.row.text,
                            "ref_audio": result.row.ref_audio_path,
                            "error": result.error_message or "Unknown error",
                        }
                    )

            batch_runtime_sec = time.perf_counter() - batch_start_time
            if batch_success_count > 0:
                batch_rtf = safe_rtf(batch_runtime_sec, batch_audio_sec)
                batch_avg_time_sec = batch_runtime_sec / batch_success_count
                batch_avg_audio_sec = batch_audio_sec / batch_success_count
                batch_avg_rtf = safe_rtf(batch_avg_time_sec, batch_avg_audio_sec)
                print(
                    f"[batch {batch_index}] size={len(row_batch)}, success={batch_success_count}, "
                    f"time_sec={batch_runtime_sec:.3f}, audio_sec={batch_audio_sec:.3f}, "
                    f"rtf={batch_rtf:.4f}, avg_time_sec={batch_avg_time_sec:.3f}, "
                    f"avg_audio_sec={batch_avg_audio_sec:.3f}, avg_rtf={batch_avg_rtf:.4f}"
                )
                total_runtime_sec += batch_runtime_sec
                total_audio_sec += batch_audio_sec
                successful_batch_count += 1
    finally:
        save_executor.shutdown(wait=True)

    save_wait_start_time = time.perf_counter()
    for _, save_future in sorted(save_futures, key=lambda item: item[0]):
        metadata_rows.append(save_future.result())
    save_wait_sec = time.perf_counter() - save_wait_start_time
    total_runtime_sec += save_wait_sec

    write_metadata(output_tsv_path, metadata_rows)
    write_failures(failed_tsv_path, failure_rows)

    overall_rtf = safe_rtf(total_runtime_sec, total_audio_sec)
    overall_avg_time_sec = total_runtime_sec / len(metadata_rows) if metadata_rows else 0.0
    overall_avg_audio_sec = total_audio_sec / len(metadata_rows) if metadata_rows else 0.0
    overall_avg_batch_time_sec = (
        total_runtime_sec / successful_batch_count if successful_batch_count else 0.0
    )
    overall_avg_batch_audio_sec = (
        total_audio_sec / successful_batch_count if successful_batch_count else 0.0
    )
    overall_avg_batch_rtf = safe_rtf(overall_avg_batch_time_sec, overall_avg_batch_audio_sec)

    print(f"Input rows: {len(rows)}")
    print(f"Generated utterances: {len(metadata_rows)}")
    print(f"Failed rows: {len(failure_rows)}")
    print(
        f"Overall timing: total_time_sec={total_runtime_sec:.3f}, "
        f"total_audio_sec={total_audio_sec:.3f}, overall_rtf={overall_rtf:.4f}, "
        f"avg_time_sec={overall_avg_time_sec:.3f}, avg_audio_sec={overall_avg_audio_sec:.3f}, "
        f"save_wait_sec={save_wait_sec:.3f}, "
        f"avg_batch_time_sec={overall_avg_batch_time_sec:.3f}, "
        f"avg_batch_audio_sec={overall_avg_batch_audio_sec:.3f}, "
        f"avg_batch_rtf={overall_avg_batch_rtf:.4f}"
    )
    print(f"WAV directory: {wav_dir}")
    print(f"Metadata TSV: {output_tsv_path}")
    print(f"Failure TSV: {failed_tsv_path}")


if __name__ == "__main__":
    main()
