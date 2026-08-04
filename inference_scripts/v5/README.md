# CosyVoice v5 Transcript-Conditioned Voice Cloning

This directory provides staged batch inference for CosyVoice 2 and CosyVoice 3.
Every reference audio must have an aligned transcript so the LLM can use the
official zero-shot voice-cloning prompt path.

## Input JSON

The JSON launcher accepts a top-level list. `reference_audio_path` and
`reference_audio_text` must be non-empty aligned lists of equal length.

```json
[
  {
    "id": "dialog-001",
    "text": [
      "Hello, thanks for calling.",
      "Can I help you with anything else?"
    ],
    "reference_audio_path": [
      "/path/to/ref_a.wav",
      "/path/to/ref_b.wav"
    ],
    "reference_audio_text": [
      "Exact transcript of reference A.",
      "Exact transcript of reference B."
    ]
  }
]
```

Each target text is paired with every aligned reference. Two texts and two
references therefore produce four candidates.

## Preparing JSON

`prepare_cosyvoice_tsv_v5.py` retains the historical name but writes JSON.
It reads each JSONL audio path and transcript as one inseparable pair.

```bash
python3 ../infer_tools/prepare_cosyvoice_tsv_v5.py \
  --text-file /path/to/targets.txt \
  --audio-jsonl /path/to/references.jsonl \
  --audio-key audio_filepath \
  --audio-text-key text \
  --references-per-text 1 \
  --output-json /path/to/input.json
```

Relative audio paths are resolved against the JSONL directory by default. Use
`--audio-root` when the manifest paths are relative to another directory.
Selection is random without replacement by default and reproducible with
`--seed`; use `--no-shuffle` for sequential JSONL order.

## Model Selection

`--model_version` is required and is validated against the model directory.

### CosyVoice 2

CosyVoice 2 uses its language tokens on target text. Reference transcripts and
reference speech tokens are supplied to the zero-shot LLM prompt.

```bash
python3 cosyvoice_generate_from_json_sbatch.py \
  --model_path /path/to/CosyVoice2-0.5B \
  --model_version cosy2 \
  --input_json /path/to/input.json \
  --output_dir /path/to/output_cosy2 \
  --lang zh \
  --batch_size 4 \
  --num_gpus 1 \
  --launcher local
```

### CosyVoice 3

CosyVoice 3 does not use `--lang`. The default system prompt is:

```text
You are a helpful assistant.<|endofprompt|>
```

The system prompt is concatenated with each reference transcript. Override it
with `--system_prompt`; custom values must end with `<|endofprompt|>`.

```bash
python3 cosyvoice_generate_from_json_sbatch.py \
  --model_path /path/to/CosyVoice3-0.5B \
  --model_version cosy3 \
  --input_json /path/to/input.json \
  --output_dir /path/to/output_cosy3 \
  --batch_size 4 \
  --num_gpus 1 \
  --launcher local
```

Passing `--lang` with `--model_version cosy3` is an error.

## Slurm Launch

The JSON launcher can submit one independent job per GPU:

```bash
python3 cosyvoice_generate_from_json_sbatch.py \
  --model_path /path/to/CosyVoice3-0.5B \
  --model_version cosy3 \
  --input_json /path/to/input.json \
  --output_dir /path/to/output \
  --batch_size 4 \
  --llm_batch_size 4 \
  --flow_batch_size 4 \
  --num_gpus 4 \
  --launcher sbatch \
  --sbatch_cmd 'sbatch --wait --partition=gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=5 --mem=30GB' \
  --conda_sh /path/to/conda.sh \
  --conda_env cosy
```

When the launcher itself is already inside an `srun` allocation, use
`--launcher local --num_gpus 1` so the child worker inherits that GPU.

## Outputs

Given `--output_dir /path/to/output`, v5 writes:

```text
/path/to/output/
  generated.json
  failed.json
  generated.tsv
  failed.tsv
  expanded_manifest.json
  chunks/
    chunk_0000/
      expanded_input.tsv
      generated.tsv
      failed.tsv
      wav/
```

`generated.json` preserves aligned candidate, reference-audio, and
reference-transcript lists:

```json
[
  {
    "id": "dialog-001",
    "output": [
      {
        "text": "Hello, thanks for calling.",
        "candidate_audio_path": ["chunks/chunk_0000/wav/utt_000000.wav"],
        "reference_audio_path": ["/path/to/ref_a.wav"],
        "reference_audio_text": ["Exact transcript of reference A."]
      }
    ]
  }
]
```

## Resume and Errors

Rerunning with the same output directory resumes by counting completed rows in
the generated and failed TSV files. Use `--overwrite` to regenerate all rows.

`--on_error skip` records row failures and continues. `--on_error raise` stops
the worker at the first failing stage. Failure output includes the target text,
reference path, reference transcript, and error message.

## Validation

Run the focused v5 tests and syntax checks from the repository root:

```bash
python3 -m unittest discover -s inference_scripts/v5/tests -v
python3 -m py_compile \
  inference_scripts/v5/*.py \
  inference_scripts/infer_tools/prepare_cosyvoice_tsv_v5.py
```
