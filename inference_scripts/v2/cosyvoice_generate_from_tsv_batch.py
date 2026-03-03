#!/usr/bin/env python3
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

import torchaudio

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from batch_types import PreparedRow, SynthesisResult, TsvInputRow
from frontend_batch import CrossLingualBatchPreparer
from io_utils import LANG_TOKEN_MAP, chunked, load_rows, write_failures, write_metadata
from staged_inference import StagedBatchInferenceRunner

DEBUG_LOG_PATH = Path("/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/.cursor/debug-e6d17b.log")
DEBUG_SESSION_ID = "e6d17b"
DEBUG_SERVER_ENDPOINT = "http://127.0.0.1:7686/ingest/409ba5f4-ff70-4548-96c8-a6f2ad82f1ac"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return
    except Exception:
        pass
    try:
        request = urllib.request.Request(
            DEBUG_SERVER_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Debug-Session-Id": DEBUG_SESSION_ID,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1):
            pass
    except (urllib.error.URLError, TimeoutError, ValueError):
        pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prepare_import_path() -> None:
    root = repo_root()
    cosyvoice_root = root / "CosyVoice"
    matcha_root = cosyvoice_root / "third_party" / "Matcha-TTS"
    sys.path.insert(0, str(cosyvoice_root))
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

    prepare_import_path()
    from cosyvoice.cli.cosyvoice import AutoModel  # pylint: disable=import-outside-toplevel
    import hyperpyyaml  # pylint: disable=import-outside-toplevel
    import ruamel.yaml  # pylint: disable=import-outside-toplevel
    matcha_root = repo_root() / "CosyVoice" / "third_party" / "Matcha-TTS"

    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H1",
        location="cosyvoice_generate_from_tsv_batch.py:main:env_probe",
        message="yaml/hyperpyyaml environment probe",
        data={
            "hyperpyyaml_version": getattr(hyperpyyaml, "__version__", "unknown"),
            "ruamel_yaml_version": getattr(ruamel.yaml, "__version__", "unknown"),
            "has_loader_max_depth": hasattr(ruamel.yaml.Loader, "max_depth"),
            "model_path": args.model_path,
            "matcha_root_exists": matcha_root.exists(),
            "matcha_root_entries": len(list(matcha_root.glob("*"))) if matcha_root.exists() else -1,
        },
    )
    # endregion
    if not hasattr(ruamel.yaml.Loader, "max_depth"):
        ruamel.yaml.Loader.max_depth = None
    if hasattr(ruamel.yaml, "SafeLoader") and not hasattr(ruamel.yaml.SafeLoader, "max_depth"):
        ruamel.yaml.SafeLoader.max_depth = None
    if hasattr(ruamel.yaml, "FullLoader") and not hasattr(ruamel.yaml.FullLoader, "max_depth"):
        ruamel.yaml.FullLoader.max_depth = None
    if hasattr(ruamel.yaml, "UnsafeLoader") and not hasattr(ruamel.yaml.UnsafeLoader, "max_depth"):
        ruamel.yaml.UnsafeLoader.max_depth = None
    # region agent log
    _debug_log(
        run_id="post-fix",
        hypothesis_id="H1",
        location="cosyvoice_generate_from_tsv_batch.py:main:yaml_monkey_patch",
        message="applied ruamel loader max_depth compatibility patch",
        data={
            "loader_has_max_depth_after_patch": hasattr(ruamel.yaml.Loader, "max_depth"),
            "loader_max_depth_value": getattr(ruamel.yaml.Loader, "max_depth", "missing"),
        },
    )
    # endregion

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
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="cosyvoice_generate_from_tsv_batch.py:main:automodel_exception",
            message="AutoModel construction failed",
            data={"error": str(exc), "error_type": exc.__class__.__name__},
        )
        # endregion
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
    failure_rows: List[Dict[str, str]] = []

    for row_batch in chunked(rows, args.batch_size):
        prepared_rows, frontend_failures = prepare_batch_rows(
            row_batch=row_batch,
            preparer=preparer,
            on_error=args.on_error,
        )
        failure_rows.extend(frontend_failures)
        if not prepared_rows:
            continue

        for result in runner.run_batch(prepared_rows):
            if result.is_success:
                metadata_rows.append(
                    save_result(
                        result=result,
                        wav_dir=wav_dir,
                        sample_rate=cosyvoice.sample_rate,
                    )
                )
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

    write_metadata(output_tsv_path, metadata_rows)
    write_failures(failed_tsv_path, failure_rows)

    print(f"Input rows: {len(rows)}")
    print(f"Generated utterances: {len(metadata_rows)}")
    print(f"Failed rows: {len(failure_rows)}")
    print(f"WAV directory: {wav_dir}")
    print(f"Metadata TSV: {output_tsv_path}")
    print(f"Failure TSV: {failed_tsv_path}")


if __name__ == "__main__":
    main()
