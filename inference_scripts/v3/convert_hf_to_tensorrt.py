#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "inference_scripts").is_dir() and (parent / "runtime").is_dir():
            return parent
    raise RuntimeError("Unable to locate project root containing inference_scripts/ and runtime/")


def run_command(command: list[str]) -> None:
    rendered = " ".join(str(token) for token in command)
    print(f"[run] {rendered}")
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HF LLM checkpoint to TensorRT-LLM weights and build TensorRT engines."
    )
    parser.add_argument("--hf_model_dir", required=True, help="HF model directory for LLM checkpoint")
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output directory containing trt_weights/ and trt_engines/",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32", "auto"],
        help="TensorRT-LLM dtype used for conversion/build (default: bfloat16)",
    )
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size (default: 1)")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size (default: 1)")
    parser.add_argument("--cp_size", type=int, default=1, help="Context parallel size (default: 1)")
    parser.add_argument("--workers", type=int, default=1, help="Checkpoint conversion workers (default: 1)")
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=16,
        help="Max TensorRT engine batch size (default: 16)",
    )
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=32768,
        help="Max TensorRT engine num tokens (default: 32768)",
    )
    parser.add_argument(
        "--skip_engine_build",
        action="store_true",
        help="Only convert HF checkpoint to TensorRT-LLM checkpoint, skip trtllm-build",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output_root before running",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tp_size <= 0 or args.pp_size <= 0 or args.cp_size <= 0:
        raise ValueError("--tp_size/--pp_size/--cp_size must be positive")
    if args.workers <= 0 or args.max_batch_size <= 0 or args.max_num_tokens <= 0:
        raise ValueError("--workers/--max_batch_size/--max_num_tokens must be positive")

    root = repo_root()
    convert_script = root / "runtime" / "triton_trtllm" / "scripts" / "convert_checkpoint.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"convert_checkpoint.py not found: {convert_script}")

    trtllm_build = shutil.which("trtllm-build")
    if trtllm_build is None and not args.skip_engine_build:
        raise RuntimeError("`trtllm-build` not found in PATH")

    hf_model_dir = Path(args.hf_model_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    trt_weights_dir = output_root / "trt_weights"
    trt_engines_dir = output_root / "trt_engines"
    if not hf_model_dir.exists():
        raise FileNotFoundError(f"HF model directory not found: {hf_model_dir}")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    convert_command = [
        sys.executable,
        str(convert_script),
        "--model_dir",
        str(hf_model_dir),
        "--output_dir",
        str(trt_weights_dir),
        "--dtype",
        args.dtype,
        "--tp_size",
        str(args.tp_size),
        "--pp_size",
        str(args.pp_size),
        "--cp_size",
        str(args.cp_size),
        "--workers",
        str(args.workers),
    ]
    run_command(convert_command)

    if args.skip_engine_build:
        print(f"[done] converted checkpoint saved at: {trt_weights_dir}")
        return

    build_command = [
        trtllm_build,
        "--checkpoint_dir",
        str(trt_weights_dir),
        "--output_dir",
        str(trt_engines_dir),
        "--max_batch_size",
        str(args.max_batch_size),
        "--max_num_tokens",
        str(args.max_num_tokens),
        "--gemm_plugin",
        args.dtype,
    ]
    run_command(build_command)
    print(f"[done] TensorRT engine directory: {trt_engines_dir}")


if __name__ == "__main__":
    main()
