# CosyVoice JSON Candidate Inference

This directory contains a JSON launcher for generating multiple candidates per
conversation turn:

```text
cosyvoice_generate_from_json_sbatch.py
```

The launcher expands each input record's `text[] x reference_audio_path[]`
combinations into flat TSV rows, runs the existing batch generator on GPU chunk
jobs, then reconstructs grouped JSON output.

The TSV batch worker remains available internally:

```text
cosyvoice_generate_from_tsv_batch.py
```

## Base Directory

Run commands from this directory:

```bash
cd /home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/inference_scripts/v4
```

## Input JSON

The input file must be a JSON list. Each object has:

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
    ]
  }
]
```

For each record, every text turn is paired with every reference audio. If one
record has 4 text turns and 5 reference audios, the launcher expands it into 20
generation candidates.

## Quick Start

```bash
python3 cosyvoice_generate_from_json_sbatch.py \
  --model_path /path/to/CosyVoice2-0.5B \
  --input_json /path/to/input.json \
  --output_dir /path/to/output_dir \
  --lang en \
  --batch_size 4 \
  --num_gpus 4 \
  --launcher sbatch \
  --sbatch_cmd 'sbatch --wait --gres=gpu:1 --ntasks=1 --cpus-per-task=8' \
  --conda_sh /mnt/users/jinyang_wang/miniforge3/etc/profile.d/conda.sh \
  --conda_env cosy
```

`--batch_size`, `--llm_batch_size`, and `--flow_batch_size` are per GPU chunk
job, not global across all GPUs.

## Outputs

Given:

```bash
--output_dir /path/to/output_dir
```

the launcher creates:

```text
/path/to/output_dir/
  generated.json
  failed.json
  generated.tsv
  failed.tsv
  expanded_manifest.json
  chunks/
    chunk_0000/
      expanded_input.tsv
      run_chunk.sh
      launch_command.txt
      submit_stdout.log
      submit_stderr.log
      stdout.log
      stderr.log
      generated.tsv
      failed.tsv
      wav/
        utt_000000.wav
    chunk_0001/
      ...
```

`generated.json` follows the grouped input shape:

```json
[
  {
    "id": "dialog-001",
    "output": [
      {
        "text": "Hello, thanks for calling.",
        "candidate_audio_path": [
          "chunks/chunk_0000/wav/utt_000000.wav",
          "chunks/chunk_0000/wav/utt_000001.wav"
        ],
        "reference_audio_path": [
          "/path/to/ref_a.wav",
          "/path/to/ref_b.wav"
        ]
      }
    ]
  }
]
```

The candidate and reference lists are aligned and follow the input reference
audio order. While jobs are still running, not-yet-generated or failed
candidates appear as `null` in `candidate_audio_path`. Failure details are
written to `failed.json` and `failed.tsv`.

`generated.tsv` and `failed.tsv` are flat merged files keyed by internal
candidate ids. They are refreshed while chunk jobs are running, and each chunk
also appends its own TSV rows after every completed batch.

`expanded_manifest.json` records the exact mapping from internal candidate ids
back to input record id, text index, and reference index.

## Resume Behavior

Rerunning the same command with the same `--output_dir` resumes automatically.
Each chunk counts existing rows in its `generated.tsv` and `failed.tsv`, treats
those rows as finished, and starts from the next input row. Chunks whose output
rows already cover all candidates are skipped by the JSON launcher.

Use `--overwrite` to ignore existing chunk outputs and regenerate from the
first candidate.

## Important Options

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `--model_path` | yes | none | Local CosyVoice model directory. |
| `--input_json` | yes | none | JSON file with `id`, `text[]`, and `reference_audio_path[]`. |
| `--output_dir` | yes | none | Directory for grouped JSON, merged TSVs, and chunk outputs. |
| `--output_json_name` | no | `generated.json` | Grouped JSON filename inside `output_dir`. |
| `--failed_json_name` | no | `failed.json` | Failure JSON filename inside `output_dir`. |
| `--output_tsv_name` | no | `generated.tsv` | Flat merged generated TSV filename. |
| `--failed_tsv_name` | no | `failed.tsv` | Flat merged failed TSV filename. |
| `--lang` | no | none | Adds language token to text: `zh`, `en`, `ja`, `yue`, or `ko`. Recommended. |
| `--text_frontend` | no | disabled | Enable CosyVoice text frontend normalization. |
| `--batch_size` | no | `4` | Expanded candidate rows processed per batch on each GPU job. |
| `--llm_batch_size` | no | `batch_size` | Micro-batch size for the LLM stage. |
| `--flow_batch_size` | no | `batch_size` | Micro-batch size for flow and vocoder stage. |
| `--flow_n_timesteps` | no | `10` | Flow diffusion steps. Lower is faster but may reduce quality. |
| `--min_token_text_ratio` | no | `2.0` | Minimum generated speech-token/text-token ratio. |
| `--max_token_text_ratio` | no | `20.0` | Maximum generated speech-token/text-token ratio. |
| `--on_error` | no | `skip` | Use `skip` to continue after row failures, or `raise` to stop. |
| `--save_workers` | no | `4` | Number of threads used to save WAV files per chunk. |
| `--num_gpus` | yes | none | Number of independent GPU chunk jobs to launch. |
| `--launcher` | no | `sbatch` | Use `sbatch` for independent GPU jobs, or `local` for direct local subprocesses. |
| `--sbatch_cmd` | no | `sbatch --wait --gres=gpu:1 --ntasks=1` | Command prefix used for each GPU chunk job. |
| `--conda_sh` | no | none | Writes `source <path>` into every chunk script before inference. |
| `--conda_env` | no | none | Writes `conda activate <env>` into every chunk script before inference. |
| `--setup_cmd` | no | none | Extra shell setup line. Can be repeated. |
| `--python_cmd` | no | `python3` | Python command used after setup commands. |
| `--chunk_dir_name` | no | `chunks` | Subdirectory under `output_dir` for per-chunk work directories. |
| `--overwrite` | no | disabled | Ignore existing chunk output TSVs and regenerate from the first candidate. |
| `--dry_run` | no | disabled | Create expanded inputs/scripts and print launch commands without running jobs. |

## Recommended Launch

```bash
cd /home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/inference_scripts/v4

python3 cosyvoice_generate_from_json_sbatch.py \
  --model_path /home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/pretrained_models/CosyVoice2-0.5B \
  --input_json /home/jinyang_wang/data/input.json \
  --output_dir /home/jinyang_wang/data/cosyvoice_outputs \
  --output_json_name generated.json \
  --failed_json_name failed.json \
  --lang en \
  --batch_size 4 \
  --llm_batch_size 4 \
  --flow_batch_size 4 \
  --flow_n_timesteps 10 \
  --on_error skip \
  --save_workers 4 \
  --num_gpus 4 \
  --launcher sbatch \
  --sbatch_cmd 'sbatch --wait --gres=gpu:1 --ntasks=1 --cpus-per-task=8' \
  --conda_sh /mnt/users/jinyang_wang/miniforge3/etc/profile.d/conda.sh \
  --conda_env cosy
```
