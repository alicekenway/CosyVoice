#!/usr/bin/env python3
import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from io_utils import (  # pylint: disable=wrong-import-position
    LANG_TOKEN_MAP,
    ORIGINAL_ROW_ID_COLUMN,
    count_tsv_rows,
    write_failures,
    write_metadata,
)


TEXT_COLUMN = "text"
REFERENCE_AUDIO_COLUMN = "reference_audio_path"
INTERNAL_ID_COLUMN = "id"


@dataclass(frozen=True)
class InputConversation:
    index: int
    input_id: str
    texts: List[str]
    reference_audio_paths: List[str]


@dataclass(frozen=True)
class ExpandedCandidate:
    candidate_id: str
    input_index: int
    input_id: str
    text_index: int
    reference_index: int
    text: str
    reference_audio_path: str


@dataclass
class ChunkJob:
    index: int
    chunk_dir: Path
    candidate_count: int
    input_tsv_path: Path
    run_script_path: Path
    stdout_path: Path
    stderr_path: Path
    submit_stdout_path: Path
    submit_stderr_path: Path
    launch_command_path: Path
    process: subprocess.Popen | None = None
    returncode: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand conversation JSON into text/reference candidates, launch "
            "single-GPU CosyVoice chunk jobs, and write grouped JSON output."
        )
    )

    parser.add_argument("--model_path", required=True, help="Local model directory path")
    parser.add_argument(
        "--input_json",
        required=True,
        help=(
            "JSON list with objects containing id, text list, and "
            "reference_audio_path list"
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for grouped JSON, merged TSVs, and chunk dirs",
    )
    parser.add_argument(
        "--output_json_name",
        default="generated.json",
        help="Name of grouped output JSON inside output_dir",
    )
    parser.add_argument(
        "--failed_json_name",
        default="failed.json",
        help="Name of grouped failure JSON inside output_dir",
    )
    parser.add_argument(
        "--output_tsv_name",
        default="generated.tsv",
        help="Name of merged flat generated TSV inside output_dir",
    )
    parser.add_argument(
        "--failed_tsv_name",
        default="failed.tsv",
        help="Name of merged flat failed-row TSV inside output_dir",
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
        help="Number of expanded text/reference rows processed per batch on each GPU",
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
        help="Error policy for row-level failures inside each chunk",
    )
    parser.add_argument(
        "--save_workers",
        type=int,
        default=4,
        help="Number of threads used to save wav files per chunk",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        required=True,
        help="Number of GPU chunk jobs to launch",
    )
    parser.add_argument(
        "--launcher",
        choices=["sbatch", "local"],
        default="sbatch",
        help=(
            "How to launch each chunk. sbatch submits independent GPU jobs; "
            "local runs chunk scripts directly for debugging."
        ),
    )
    parser.add_argument(
        "--sbatch_cmd",
        default="sbatch --wait --gres=gpu:1 --ntasks=1",
        help=(
            "sbatch command prefix for each chunk, quoted as one string. "
            "The wrapper adds per-chunk --output/--error paths and the chunk "
            "script path. --wait is added automatically if omitted."
        ),
    )
    parser.add_argument(
        "--setup_cmd",
        action="append",
        default=[],
        help=(
            "Bash setup line inserted before the chunk command. Can be repeated, "
            "for example: --setup_cmd 'source .../conda.sh' "
            "--setup_cmd 'conda activate cosy'."
        ),
    )
    parser.add_argument(
        "--conda_sh",
        default=None,
        help="Optional path to conda.sh; writes 'source <path>' in each chunk script",
    )
    parser.add_argument(
        "--conda_env",
        default=None,
        help="Optional conda environment name; writes 'conda activate <env>'",
    )
    parser.add_argument(
        "--python_cmd",
        default="python3",
        help="Python executable/command used after setup commands",
    )
    parser.add_argument(
        "--chunk_dir_name",
        default="chunks",
        help="Subdirectory under output_dir used for per-chunk work dirs",
    )
    parser.add_argument(
        "--poll_interval_sec",
        type=float,
        default=5.0,
        help="Seconds between chunk status polls",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Create expanded inputs/scripts and print launch commands without running jobs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Ignore existing chunk output TSVs and regenerate from the first "
            "candidate in each chunk. By default, existing generated/failed "
            "TSV rows are treated as finished rows for resume."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_gpus <= 0:
        raise ValueError("--num_gpus must be positive")
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
    if args.poll_interval_sec <= 0:
        raise ValueError("--poll_interval_sec must be positive")
    if not args.python_cmd.strip():
        raise ValueError("--python_cmd must not be empty")


def require_string_list(value, record_index: int, field_name: str) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"Record {record_index} field '{field_name}' must be a list")
    output: List[str] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(
                f"Record {record_index} field '{field_name}' item {item_index} must be a string"
            )
        stripped_item = item.strip()
        if not stripped_item:
            raise ValueError(
                f"Record {record_index} field '{field_name}' item {item_index} is empty"
            )
        output.append(stripped_item)
    if not output:
        raise ValueError(f"Record {record_index} field '{field_name}' must not be empty")
    return output


def load_input_conversations(input_json_path: Path) -> List[InputConversation]:
    with input_json_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list of conversation objects")

    conversations: List[InputConversation] = []
    for record_index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"Record {record_index} must be an object")
        input_id = record.get("id", "")
        if input_id is None:
            input_id = ""
        if not isinstance(input_id, str):
            input_id = str(input_id)
        texts = require_string_list(record.get("text"), record_index, "text")
        reference_audio_paths = require_string_list(
            record.get("reference_audio_path"),
            record_index,
            "reference_audio_path",
        )
        conversations.append(
            InputConversation(
                index=record_index,
                input_id=input_id,
                texts=texts,
                reference_audio_paths=reference_audio_paths,
            )
        )
    if not conversations:
        raise ValueError(f"No records found in input JSON: {input_json_path}")
    return conversations


def candidate_id(input_index: int, text_index: int, reference_index: int) -> str:
    return f"rec{input_index:06d}_text{text_index:06d}_ref{reference_index:06d}"


def expand_conversations(conversations: Sequence[InputConversation]) -> List[ExpandedCandidate]:
    candidates: List[ExpandedCandidate] = []
    for conversation in conversations:
        for text_index, text in enumerate(conversation.texts):
            for reference_index, reference_audio_path in enumerate(
                conversation.reference_audio_paths
            ):
                candidates.append(
                    ExpandedCandidate(
                        candidate_id=candidate_id(
                            conversation.index,
                            text_index,
                            reference_index,
                        ),
                        input_index=conversation.index,
                        input_id=conversation.input_id,
                        text_index=text_index,
                        reference_index=reference_index,
                        text=text,
                        reference_audio_path=reference_audio_path,
                    )
                )
    return candidates


def candidate_tsv_row(candidate: ExpandedCandidate) -> Dict[str, str]:
    return {
        INTERNAL_ID_COLUMN: candidate.candidate_id,
        TEXT_COLUMN: candidate.text,
        REFERENCE_AUDIO_COLUMN: candidate.reference_audio_path,
        ORIGINAL_ROW_ID_COLUMN: candidate.candidate_id,
    }


def split_items(items: Sequence[ExpandedCandidate], requested_chunks: int) -> List[List[ExpandedCandidate]]:
    chunk_count = min(requested_chunks, len(items))
    if chunk_count <= 0:
        raise ValueError("No expanded text/reference candidates to generate")
    base_size, remainder = divmod(len(items), chunk_count)
    chunks: List[List[ExpandedCandidate]] = []
    start_index = 0
    for chunk_index in range(chunk_count):
        size = base_size + (1 if chunk_index < remainder else 0)
        end_index = start_index + size
        chunks.append(list(items[start_index:end_index]))
        start_index = end_index
    return chunks


def write_chunk_tsv(chunk_input_path: Path, candidates: Sequence[ExpandedCandidate]) -> None:
    fieldnames = [
        INTERNAL_ID_COLUMN,
        TEXT_COLUMN,
        REFERENCE_AUDIO_COLUMN,
        ORIGINAL_ROW_ID_COLUMN,
    ]
    with chunk_input_path.open("w", encoding="utf-8", newline="") as chunk_file:
        writer = csv.DictWriter(chunk_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(candidate_tsv_row(candidate) for candidate in candidates)


def write_manifest(
    manifest_path: Path,
    conversations: Sequence[InputConversation],
    candidates: Sequence[ExpandedCandidate],
) -> None:
    payload = {
        "input": [asdict(conversation) for conversation in conversations],
        "expanded": [asdict(candidate) for candidate in candidates],
    }
    write_json(manifest_path, payload)


def setup_lines(args: argparse.Namespace) -> List[str]:
    lines: List[str] = []
    if args.conda_sh:
        conda_sh_path = Path(args.conda_sh).expanduser()
        lines.append(f"source {shlex.quote(str(conda_sh_path))}")
    if args.conda_env:
        lines.append(f"conda activate {shlex.quote(args.conda_env)}")
    lines.extend(args.setup_cmd or [])
    return lines


def child_command(args: argparse.Namespace, chunk_input_path: Path, chunk_dir: Path) -> List[str]:
    llm_batch_size = args.llm_batch_size or args.batch_size
    flow_batch_size = args.flow_batch_size or args.batch_size
    command = [
        *shlex.split(args.python_cmd),
        str(CURRENT_DIR / "cosyvoice_generate_from_tsv_batch.py"),
        "--model_path",
        str(Path(args.model_path).expanduser().resolve()),
        "--input_tsv",
        str(chunk_input_path),
        "--output_dir",
        str(chunk_dir),
        "--output_tsv_name",
        args.output_tsv_name,
        "--failed_tsv_name",
        args.failed_tsv_name,
        "--batch_size",
        str(args.batch_size),
        "--llm_batch_size",
        str(llm_batch_size),
        "--flow_batch_size",
        str(flow_batch_size),
        "--min_token_text_ratio",
        str(args.min_token_text_ratio),
        "--max_token_text_ratio",
        str(args.max_token_text_ratio),
        "--flow_n_timesteps",
        str(args.flow_n_timesteps),
        "--on_error",
        args.on_error,
        "--save_workers",
        str(args.save_workers),
    ]
    if args.text_frontend:
        command.append("--text_frontend")
    if args.lang:
        command.extend(["--lang", args.lang])
    if args.overwrite:
        command.append("--overwrite")
    return command


def write_run_script(
    run_script_path: Path,
    setup_commands: Sequence[str],
    command: Sequence[str],
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -eo pipefail",
        f"cd {shlex.quote(str(CURRENT_DIR))}",
    ]
    lines.extend(setup_commands)
    lines.append(f"exec {shlex.join(command)}")
    run_script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_script_path.chmod(0o755)


def build_jobs(
    args: argparse.Namespace,
    chunks: Sequence[Sequence[ExpandedCandidate]],
    output_dir: Path,
) -> List[ChunkJob]:
    chunk_root = output_dir / args.chunk_dir_name
    chunk_root.mkdir(parents=True, exist_ok=True)
    jobs: List[ChunkJob] = []
    commands = setup_lines(args)
    for chunk_index, candidates in enumerate(chunks):
        chunk_dir = chunk_root / f"chunk_{chunk_index:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        input_tsv_path = chunk_dir / "expanded_input.tsv"
        run_script_path = chunk_dir / "run_chunk.sh"
        stdout_path = chunk_dir / "stdout.log"
        stderr_path = chunk_dir / "stderr.log"
        submit_stdout_path = chunk_dir / "submit_stdout.log"
        submit_stderr_path = chunk_dir / "submit_stderr.log"
        launch_command_path = chunk_dir / "launch_command.txt"

        write_chunk_tsv(input_tsv_path, candidates)
        command = child_command(args, input_tsv_path, chunk_dir)
        write_run_script(run_script_path, commands, command)

        jobs.append(
            ChunkJob(
                index=chunk_index,
                chunk_dir=chunk_dir,
                candidate_count=len(candidates),
                input_tsv_path=input_tsv_path,
                run_script_path=run_script_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                submit_stdout_path=submit_stdout_path,
                submit_stderr_path=submit_stderr_path,
                launch_command_path=launch_command_path,
            )
        )
    return jobs


def completed_row_count_for_job(
    job: ChunkJob,
    output_tsv_name: str,
    failed_tsv_name: str,
) -> int:
    return count_tsv_rows(job.chunk_dir / output_tsv_name) + count_tsv_rows(
        job.chunk_dir / failed_tsv_name
    )


def jobs_needing_launch(
    jobs: Sequence[ChunkJob],
    output_tsv_name: str,
    failed_tsv_name: str,
    overwrite: bool,
) -> List[ChunkJob]:
    if overwrite:
        return list(jobs)

    pending_jobs: List[ChunkJob] = []
    for job in jobs:
        completed_count = completed_row_count_for_job(
            job,
            output_tsv_name,
            failed_tsv_name,
        )
        if completed_count > job.candidate_count:
            raise ValueError(
                f"{job.chunk_dir} has {completed_count} finished rows, but "
                f"this chunk only has {job.candidate_count} candidates. Use "
                "--overwrite or a clean output directory if these files are stale."
            )
        if completed_count == job.candidate_count:
            job.returncode = 0
            print(
                f"[resume chunk {job.index}] already complete "
                f"({completed_count}/{job.candidate_count}); skipping launch"
            )
            continue
        if completed_count:
            print(
                f"[resume chunk {job.index}] {completed_count}/"
                f"{job.candidate_count} rows finished; launching for remainder"
            )
        pending_jobs.append(job)
    return pending_jobs


def split_command(command: str) -> List[str]:
    if not command.strip():
        return []
    return shlex.split(command)


def has_long_option(tokens: Sequence[str], option_name: str) -> bool:
    return any(token == option_name or token.startswith(f"{option_name}=") for token in tokens)


def sbatch_prefix(sbatch_cmd: str) -> List[str]:
    prefix = split_command(sbatch_cmd)
    if not prefix:
        raise ValueError("--sbatch_cmd must not be empty when --launcher sbatch is used")
    if not has_long_option(prefix, "--wait"):
        prefix.append("--wait")
    return prefix


def launch_command_for_job(args: argparse.Namespace, job: ChunkJob) -> List[str]:
    if args.launcher == "sbatch":
        prefix = sbatch_prefix(args.sbatch_cmd)
        if not has_long_option(prefix, "--job-name"):
            prefix.extend(["--job-name", f"cosyvoice_v4_chunk_{job.index:04d}"])
        return [
            *prefix,
            "--output",
            str(job.stdout_path),
            "--error",
            str(job.stderr_path),
            str(job.run_script_path),
        ]
    return [str(job.run_script_path)]


def launch_log_paths(job: ChunkJob, launcher: str) -> tuple[Path, Path]:
    if launcher == "sbatch":
        return job.submit_stdout_path, job.submit_stderr_path
    return job.stdout_path, job.stderr_path


def launch_jobs(args: argparse.Namespace, jobs: Sequence[ChunkJob]) -> None:
    launched_jobs: List[ChunkJob] = []
    try:
        for job in jobs:
            launch_command = launch_command_for_job(args, job)
            job.launch_command_path.write_text(
                shlex.join(launch_command) + "\n",
                encoding="utf-8",
            )
            print(f"[launch chunk {job.index}] {shlex.join(launch_command)}")
            if args.dry_run:
                continue
            stdout_path, stderr_path = launch_log_paths(job, args.launcher)
            with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_file:
                job.process = subprocess.Popen(
                    launch_command,
                    cwd=str(job.chunk_dir),
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
            launched_jobs.append(job)
    except Exception:
        for launched_job in launched_jobs:
            if launched_job.process is not None and launched_job.process.poll() is None:
                launched_job.process.terminate()
        for launched_job in launched_jobs:
            if launched_job.process is not None:
                launched_job.process.wait()
        raise


def wait_for_jobs(
    jobs: Sequence[ChunkJob],
    poll_interval_sec: float,
    poll_callback: Callable[[], None] | None = None,
) -> None:
    active_jobs = [job for job in jobs if job.process is not None]
    try:
        while active_jobs:
            for job in list(active_jobs):
                assert job.process is not None
                returncode = job.process.poll()
                if returncode is None:
                    continue
                job.returncode = returncode
                active_jobs.remove(job)
                status = "ok" if returncode == 0 else f"failed rc={returncode}"
                print(f"[finish chunk {job.index}] {status}; logs={job.chunk_dir}")
            if poll_callback is not None:
                poll_callback()
            if active_jobs:
                time.sleep(poll_interval_sec)
    except KeyboardInterrupt:
        for job in active_jobs:
            assert job.process is not None
            job.process.terminate()
        for job in active_jobs:
            assert job.process is not None
            job.process.wait()
        raise


def ensure_all_jobs_succeeded(jobs: Sequence[ChunkJob]) -> None:
    failed_jobs = [job for job in jobs if job.returncode not in (0, None)]
    if not failed_jobs:
        return
    details = ", ".join(
        f"chunk_{job.index:04d} rc={job.returncode} "
        f"stderr={job.stderr_path} submit_stderr={job.submit_stderr_path}"
        for job in failed_jobs
    )
    raise RuntimeError(f"One or more chunk jobs failed; not merging outputs: {details}")


def ensure_chunk_outputs_exist(
    jobs: Sequence[ChunkJob],
    output_tsv_name: str,
    failed_tsv_name: str,
) -> None:
    missing_paths: List[Path] = []
    for job in jobs:
        for path in (job.chunk_dir / output_tsv_name, job.chunk_dir / failed_tsv_name):
            if not path.exists():
                missing_paths.append(path)
    if missing_paths:
        joined_paths = ", ".join(str(path) for path in missing_paths)
        raise RuntimeError(f"Missing expected chunk output TSVs; not merging: {joined_paths}")


def read_tsv_rows(tsv_path: Path) -> List[Dict[str, str]]:
    if not tsv_path.exists():
        return []
    with tsv_path.open("r", encoding="utf-8", newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        if not reader.fieldnames:
            return []
        rows: List[Dict[str, str]] = []
        for row in reader:
            if None in row:
                continue
            rows.append(dict(row))
        return rows


def prefixed_speechpath(output_dir: Path, chunk_dir: Path, speechpath: str) -> str:
    relative_chunk_dir = chunk_dir.resolve().relative_to(output_dir.resolve()).as_posix()
    return f"{relative_chunk_dir}/{speechpath}" if speechpath else relative_chunk_dir


def merge_outputs(
    jobs: Sequence[ChunkJob],
    output_dir: Path,
    output_tsv_name: str,
    failed_tsv_name: str,
) -> tuple[int, int]:
    metadata_rows: List[Dict[str, str]] = []
    failure_rows: List[Dict[str, str]] = []

    for job in jobs:
        chunk_generated_path = job.chunk_dir / output_tsv_name
        chunk_failed_path = job.chunk_dir / failed_tsv_name

        for row in read_tsv_rows(chunk_generated_path):
            metadata_rows.append(
                {
                    "speechpath": prefixed_speechpath(
                        output_dir,
                        job.chunk_dir,
                        row.get("speechpath", ""),
                    ),
                    "text": row.get("text", ""),
                    "id": row.get("id", ""),
                }
            )

        for row in read_tsv_rows(chunk_failed_path):
            failure_rows.append(
                {
                    "row_id": row.get("row_id", ""),
                    "id": row.get("id", ""),
                    "text": row.get("text", ""),
                    "ref_audio": row.get("ref_audio", ""),
                    "error": row.get("error", ""),
                }
            )

    write_metadata(output_dir / output_tsv_name, metadata_rows, include_input_id=True)
    write_failures(output_dir / failed_tsv_name, failure_rows, include_input_id=True)
    return len(metadata_rows), len(failure_rows)


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def grouped_json_payload(
    conversations: Sequence[InputConversation],
    generated_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    generated_by_candidate = {
        row.get("id", ""): row.get("speechpath", "")
        for row in generated_rows
        if row.get("id")
    }

    payload: List[Dict[str, object]] = []
    for conversation in conversations:
        outputs: List[Dict[str, object]] = []
        for text_index, text in enumerate(conversation.texts):
            candidate_audio_paths: List[str | None] = []
            reference_audio_paths: List[str] = []
            for reference_index, reference_audio_path in enumerate(
                conversation.reference_audio_paths
            ):
                current_candidate_id = candidate_id(
                    conversation.index,
                    text_index,
                    reference_index,
                )
                candidate_audio_paths.append(
                    generated_by_candidate.get(current_candidate_id)
                )
                reference_audio_paths.append(reference_audio_path)
            outputs.append(
                {
                    "text": text,
                    "candidate_audio_path": candidate_audio_paths,
                    "reference_audio_path": reference_audio_paths,
                }
            )
        payload.append({"id": conversation.input_id, "output": outputs})
    return payload


def failed_json_payload(
    candidates_by_id: Dict[str, ExpandedCandidate],
    failure_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for row in failure_rows:
        row_candidate_id = row.get("id") or row.get("row_id", "")
        candidate = candidates_by_id.get(row_candidate_id)
        payload.append(
            {
                "candidate_id": row_candidate_id,
                "id": candidate.input_id if candidate is not None else "",
                "text_index": candidate.text_index if candidate is not None else None,
                "reference_index": (
                    candidate.reference_index if candidate is not None else None
                ),
                "text": row.get("text") or (candidate.text if candidate is not None else ""),
                "reference_audio_path": (
                    candidate.reference_audio_path
                    if candidate is not None
                    else row.get("ref_audio", "")
                ),
                "error": row.get("error", ""),
            }
        )
    return payload


def refresh_grouped_json_outputs(
    output_dir: Path,
    output_tsv_name: str,
    failed_tsv_name: str,
    output_json_name: str,
    failed_json_name: str,
    conversations: Sequence[InputConversation],
    candidates_by_id: Dict[str, ExpandedCandidate],
) -> tuple[int, int]:
    generated_rows = read_tsv_rows(output_dir / output_tsv_name)
    failure_rows = read_tsv_rows(output_dir / failed_tsv_name)
    write_json(
        output_dir / output_json_name,
        grouped_json_payload(conversations, generated_rows),
    )
    write_json(
        output_dir / failed_json_name,
        failed_json_payload(candidates_by_id, failure_rows),
    )
    return len(generated_rows), len(failure_rows)


def main() -> None:
    args = parse_args()
    validate_args(args)

    input_json_path = Path(args.input_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conversations = load_input_conversations(input_json_path)
    candidates = expand_conversations(conversations)
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    chunks = split_items(candidates, args.num_gpus)
    if len(chunks) < args.num_gpus:
        print(
            f"Requested {args.num_gpus} GPU chunks but only {len(candidates)} "
            f"expanded candidates exist; launching {len(chunks)} chunks."
        )

    write_manifest(output_dir / "expanded_manifest.json", conversations, candidates)
    jobs = build_jobs(args, chunks, output_dir)
    print(f"Input records: {len(conversations)}")
    print(f"Expanded candidates: {len(candidates)}")
    print(f"Chunk jobs: {len(jobs)}")
    print(f"Chunk root: {output_dir / args.chunk_dir_name}")

    launchable_jobs = jobs_needing_launch(
        jobs=jobs,
        output_tsv_name=args.output_tsv_name,
        failed_tsv_name=args.failed_tsv_name,
        overwrite=args.overwrite,
    )
    launch_jobs(args, launchable_jobs)
    if args.dry_run:
        print("Dry run complete; no chunk jobs were launched and no grouped JSON was written.")
        return

    def refresh_outputs() -> None:
        merge_outputs(
            jobs=jobs,
            output_dir=output_dir,
            output_tsv_name=args.output_tsv_name,
            failed_tsv_name=args.failed_tsv_name,
        )
        refresh_grouped_json_outputs(
            output_dir=output_dir,
            output_tsv_name=args.output_tsv_name,
            failed_tsv_name=args.failed_tsv_name,
            output_json_name=args.output_json_name,
            failed_json_name=args.failed_json_name,
            conversations=conversations,
            candidates_by_id=candidates_by_id,
        )

    refresh_outputs()
    wait_for_jobs(launchable_jobs, args.poll_interval_sec, poll_callback=refresh_outputs)
    ensure_all_jobs_succeeded(jobs)
    ensure_chunk_outputs_exist(jobs, args.output_tsv_name, args.failed_tsv_name)
    generated_count, failure_count = merge_outputs(
        jobs=jobs,
        output_dir=output_dir,
        output_tsv_name=args.output_tsv_name,
        failed_tsv_name=args.failed_tsv_name,
    )
    refresh_grouped_json_outputs(
        output_dir=output_dir,
        output_tsv_name=args.output_tsv_name,
        failed_tsv_name=args.failed_tsv_name,
        output_json_name=args.output_json_name,
        failed_json_name=args.failed_json_name,
        conversations=conversations,
        candidates_by_id=candidates_by_id,
    )

    print(f"Merged generated candidates: {generated_count}")
    print(f"Merged failed candidates: {failure_count}")
    print(f"Grouped JSON: {output_dir / args.output_json_name}")
    print(f"Failure JSON: {output_dir / args.failed_json_name}")
    print(f"Merged TSV: {output_dir / args.output_tsv_name}")
    print(f"Failure TSV: {output_dir / args.failed_tsv_name}")


if __name__ == "__main__":
    main()
