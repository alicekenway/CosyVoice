#!/usr/bin/env python3
import argparse
import csv
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from io_utils import (  # pylint: disable=wrong-import-position
    LANG_TOKEN_MAP,
    ORIGINAL_ROW_ID_COLUMN,
    normalize_header,
    resolve_columns,
    write_failures,
    write_metadata,
)


@dataclass
class ChunkJob:
    index: int
    chunk_dir: Path
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
            "Split a CosyVoice TSV inference job across multiple Slurm chunk "
            "jobs, then merge the chunk outputs."
        )
    )

    # Same inference surface as cosyvoice_generate_from_tsv_batch.py.
    parser.add_argument("--model_path", required=True, help="Local model directory path")
    parser.add_argument(
        "--input_tsv",
        required=True,
        help="TSV file with columns for text and reference audio path",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for merged TSVs and per-chunk subdirectories",
    )
    parser.add_argument(
        "--output_tsv_name",
        default="generated.tsv",
        help="Name of merged output TSV inside output_dir (default: generated.tsv)",
    )
    parser.add_argument(
        "--failed_tsv_name",
        default="failed.tsv",
        help="Name of merged failed-row TSV inside output_dir (default: failed.tsv)",
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
        help="Number of TSV rows processed together inside each chunk",
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
        help="Error policy for row-level failures inside each chunk (default: skip)",
    )
    parser.add_argument(
        "--save_workers",
        type=int,
        default=4,
        help="Number of threads used to save wav files per chunk (default: 4)",
    )

    # Launcher-only options.
    parser.add_argument(
        "--num_gpus",
        type=int,
        required=True,
        help="Number of GPU chunks/subprocesses to launch",
    )
    parser.add_argument(
        "--launcher",
        choices=["sbatch", "local"],
        default="sbatch",
        help=(
            "How to launch each chunk. sbatch submits independent GPU jobs "
            "(recommended when the master runs in a CPU-only allocation); "
            "local runs the chunk scripts directly."
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
        help="Python executable/command used after setup commands (default: python3)",
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
        help="Seconds between chunk status polls (default: 5)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Create chunk inputs/scripts and print launch commands without running jobs",
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


def has_input_id(fieldnames: Sequence[str]) -> bool:
    return any(normalize_header(field_name) == "id" for field_name in fieldnames)


def read_valid_input_rows(input_tsv_path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with input_tsv_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Input TSV has no header: {input_tsv_path}")
        fieldnames = list(reader.fieldnames)
        column_mapping = resolve_columns(fieldnames)
        if "text" not in column_mapping or "ref_audio" not in column_mapping:
            raise ValueError(
                "Input TSV must contain columns for text and reference audio path. "
                f"Found header: {reader.fieldnames}"
            )

        output_fieldnames = list(fieldnames)
        if ORIGINAL_ROW_ID_COLUMN not in output_fieldnames:
            output_fieldnames.append(ORIGINAL_ROW_ID_COLUMN)

        rows: List[Dict[str, str]] = []
        for original_row_id, row in enumerate(reader, start=1):
            text = (row.get(column_mapping["text"], "") or "").strip()
            ref_audio_path = (row.get(column_mapping["ref_audio"], "") or "").strip()
            if not text or not ref_audio_path:
                continue

            output_row = {field_name: row.get(field_name, "") or "" for field_name in fieldnames}
            output_row[ORIGINAL_ROW_ID_COLUMN] = str(original_row_id)
            rows.append(output_row)

    if not rows:
        raise ValueError(f"No valid rows found in input TSV: {input_tsv_path}")
    return output_fieldnames, rows


def split_rows(rows: Sequence[Dict[str, str]], requested_chunks: int) -> List[List[Dict[str, str]]]:
    chunk_count = min(requested_chunks, len(rows))
    base_size, remainder = divmod(len(rows), chunk_count)
    chunks: List[List[Dict[str, str]]] = []
    start_index = 0
    for chunk_index in range(chunk_count):
        size = base_size + (1 if chunk_index < remainder else 0)
        end_index = start_index + size
        chunks.append(list(rows[start_index:end_index]))
        start_index = end_index
    return chunks


def write_chunk_tsv(
    chunk_input_path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Dict[str, str]],
) -> None:
    with chunk_input_path.open("w", encoding="utf-8", newline="") as chunk_file:
        writer = csv.DictWriter(chunk_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


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
    return command


def write_run_script(
    run_script_path: Path,
    setup_commands: Sequence[str],
    chunk_command: Sequence[str],
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -eo pipefail",
        f"cd {shlex.quote(str(CURRENT_DIR))}",
    ]
    lines.extend(setup_commands)
    lines.append(f"exec {shlex.join(chunk_command)}")
    run_script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_script_path.chmod(0o755)


def build_jobs(
    args: argparse.Namespace,
    fieldnames: Sequence[str],
    chunks: Sequence[Sequence[Dict[str, str]]],
    output_dir: Path,
) -> List[ChunkJob]:
    chunk_root = output_dir / args.chunk_dir_name
    chunk_root.mkdir(parents=True, exist_ok=True)
    jobs: List[ChunkJob] = []
    commands = setup_lines(args)
    for chunk_index, rows in enumerate(chunks):
        chunk_dir = chunk_root / f"chunk_{chunk_index:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        input_tsv_path = chunk_dir / "input.tsv"
        run_script_path = chunk_dir / "run_chunk.sh"
        stdout_path = chunk_dir / "stdout.log"
        stderr_path = chunk_dir / "stderr.log"
        submit_stdout_path = chunk_dir / "submit_stdout.log"
        submit_stderr_path = chunk_dir / "submit_stderr.log"
        launch_command_path = chunk_dir / "launch_command.txt"

        write_chunk_tsv(input_tsv_path, fieldnames, rows)
        command = child_command(args, input_tsv_path, chunk_dir)
        write_run_script(run_script_path, commands, command)

        jobs.append(
            ChunkJob(
                index=chunk_index,
                chunk_dir=chunk_dir,
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
            prefix.extend(["--job-name", f"cosyvoice_chunk_{job.index:04d}"])
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
            # A launcher can read while a chunk process is appending. Skip any
            # partial trailing record rather than surfacing it in the merged TSV.
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
    include_input_id: bool,
) -> tuple[int, int]:
    metadata_rows: List[Dict[str, str]] = []
    failure_rows: List[Dict[str, str]] = []

    for job in jobs:
        chunk_generated_path = job.chunk_dir / output_tsv_name
        chunk_failed_path = job.chunk_dir / failed_tsv_name

        for row in read_tsv_rows(chunk_generated_path):
            merged_row = {
                "speechpath": prefixed_speechpath(
                    output_dir,
                    job.chunk_dir,
                    row.get("speechpath", ""),
                ),
                "text": row.get("text", ""),
            }
            if include_input_id:
                merged_row["id"] = row.get("id", "")
            metadata_rows.append(merged_row)

        for row in read_tsv_rows(chunk_failed_path):
            merged_row = {
                "row_id": row.get("row_id", ""),
                "text": row.get("text", ""),
                "ref_audio": row.get("ref_audio", ""),
                "error": row.get("error", ""),
            }
            if include_input_id:
                merged_row["id"] = row.get("id", "")
            failure_rows.append(merged_row)

    write_metadata(output_dir / output_tsv_name, metadata_rows, include_input_id=include_input_id)
    write_failures(output_dir / failed_tsv_name, failure_rows, include_input_id=include_input_id)
    return len(metadata_rows), len(failure_rows)


def main() -> None:
    args = parse_args()
    validate_args(args)

    input_tsv_path = Path(args.input_tsv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = read_valid_input_rows(input_tsv_path)
    chunks = split_rows(rows, args.num_gpus)
    if len(chunks) < args.num_gpus:
        print(
            f"Requested {args.num_gpus} GPU chunks but only {len(rows)} valid rows exist; "
            f"launching {len(chunks)} chunks."
        )

    jobs = build_jobs(args, fieldnames, chunks, output_dir)
    print(f"Valid input rows: {len(rows)}")
    print(f"Chunk jobs: {len(jobs)}")
    print(f"Chunk root: {output_dir / args.chunk_dir_name}")

    launch_jobs(args, jobs)
    if args.dry_run:
        print("Dry run complete; no chunk jobs were launched and no merged TSV was written.")
        return

    include_input_id = has_input_id(fieldnames)

    def refresh_merged_outputs() -> None:
        merge_outputs(
            jobs=jobs,
            output_dir=output_dir,
            output_tsv_name=args.output_tsv_name,
            failed_tsv_name=args.failed_tsv_name,
            include_input_id=include_input_id,
        )

    refresh_merged_outputs()
    wait_for_jobs(jobs, args.poll_interval_sec, poll_callback=refresh_merged_outputs)
    ensure_all_jobs_succeeded(jobs)
    ensure_chunk_outputs_exist(jobs, args.output_tsv_name, args.failed_tsv_name)
    generated_count, failure_count = merge_outputs(
        jobs=jobs,
        output_dir=output_dir,
        output_tsv_name=args.output_tsv_name,
        failed_tsv_name=args.failed_tsv_name,
        include_input_id=include_input_id,
    )

    print(f"Merged generated utterances: {generated_count}")
    print(f"Merged failed rows: {failure_count}")
    print(f"Metadata TSV: {output_dir / args.output_tsv_name}")
    print(f"Failure TSV: {output_dir / args.failed_tsv_name}")


if __name__ == "__main__":
    main()
