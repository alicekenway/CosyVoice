import os
from typing import Mapping


COMPUTE_MODES = ("gpu", "cpu")
DEFAULT_GPU_BATCH_SIZE = 4
DEFAULT_CPU_BATCH_SIZE = 1


def resolve_batch_sizes(
    mode: str,
    batch_size: int | None,
    llm_batch_size: int | None,
    flow_batch_size: int | None,
) -> tuple[int, int, int]:
    if mode not in COMPUTE_MODES:
        raise ValueError(f"Unsupported compute mode: {mode}")

    default_batch_size = (
        DEFAULT_CPU_BATCH_SIZE if mode == "cpu" else DEFAULT_GPU_BATCH_SIZE
    )
    resolved_batch_size = batch_size if batch_size is not None else default_batch_size
    resolved_llm_batch_size = (
        llm_batch_size if llm_batch_size is not None else resolved_batch_size
    )
    resolved_flow_batch_size = (
        flow_batch_size if flow_batch_size is not None else resolved_batch_size
    )

    named_sizes = {
        "--batch_size": resolved_batch_size,
        "--llm_batch_size": resolved_llm_batch_size,
        "--flow_batch_size": resolved_flow_batch_size,
    }
    for option_name, value in named_sizes.items():
        if value <= 0:
            raise ValueError(f"{option_name} must be positive")
    if mode == "cpu":
        non_unit_sizes = [
            f"{option_name}={value}"
            for option_name, value in named_sizes.items()
            if value != 1
        ]
        if non_unit_sizes:
            raise ValueError(
                "CPU mode only supports batch size 1; received "
                + ", ".join(non_unit_sizes)
            )
    return (
        resolved_batch_size,
        resolved_llm_batch_size,
        resolved_flow_batch_size,
    )


def resolve_cpu_threads(environment: Mapping[str, str] | None = None) -> int:
    env = os.environ if environment is None else environment
    raw_value = env.get("COSYVOICE_CPU_THREADS") or env.get("SLURM_CPUS_PER_TASK") or "1"
    try:
        thread_count = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "CPU thread count must be a positive integer; got "
            f"{raw_value!r} from COSYVOICE_CPU_THREADS/SLURM_CPUS_PER_TASK"
        ) from exc
    if thread_count <= 0:
        raise ValueError(f"CPU thread count must be positive, got {thread_count}")
    return thread_count
