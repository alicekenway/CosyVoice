# CosyVoice TSV Batch Inference

This directory contains a staged CosyVoice batch inference program:

```text
cosyvoice_generate_from_tsv_batch.py
```

It reads a TSV file containing target text and a reference audio path for each row,
generates speech with CosyVoice, writes WAV files, and emits metadata/failure TSVs.

## Base Directory

Run commands from this directory:

```bash
cd /home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/inference_scripts/v2
```

The script adds the local CosyVoice project and `third_party/Matcha-TTS` to
`PYTHONPATH` automatically.

## Quick Start

```bash
python3 cosyvoice_generate_from_tsv_batch.py \
  --model_path /path/to/CosyVoice2-0.5B \
  --input_tsv /path/to/input.tsv \
  --output_dir /path/to/output_dir \
  --lang en \
  --batch_size 4
```

For Chinese text:

```bash
python3 cosyvoice_generate_from_tsv_batch.py \
  --model_path /path/to/CosyVoice2-0.5B \
  --input_tsv /path/to/input.tsv \
  --output_dir /path/to/output_dir \
  --lang zh \
  --batch_size 4
```

## Input TSV

The TSV must have a header row and at least these two columns:

```tsv
text	reference_audio_path
Hello, this is a test.	/home/jinyang_wang/audio/prompt.wav
```

The text column must be named `text`.

An optional `id` column is also supported:

```tsv
text	reference_audio_path	id
Hello, this is a test.	/home/jinyang_wang/audio/prompt.wav	sample-001
```

The reference audio column can use any of these names:

```text
reference_audio_path
reference wav path
ref audio path
ref wav path
audio path
wav path
prompt wav
prompt wav path
prompt speech path
```

Header matching is case-insensitive and treats underscores like spaces, so
`reference_audio_path` and `Reference Audio Path` are both valid.

Rows with empty text or empty reference audio path are skipped.

## Outputs

Given:

```bash
--output_dir /path/to/output_dir
```

the script creates:

```text
/path/to/output_dir/
  wav/
    utt_000000.wav
    utt_000001.wav
    ...
  generated.tsv
  failed.tsv
```

`generated.tsv` has:

```tsv
speechpath	text
wav/utt_000000.wav	<|en|>Hello, this is a test.
```

If the input TSV has an `id` column, `generated.tsv` also includes it:

```tsv
speechpath	text	id
wav/utt_000000.wav	<|en|>Hello, this is a test.	sample-001
```

`failed.tsv` has:

```tsv
row_id	text	ref_audio	error
```

If the input TSV has an `id` column, `failed.tsv` includes it after `row_id`.

`row_id` is the input TSV data row number, starting at `1` after the header.

## Important Options

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `--model_path` | yes | none | Local CosyVoice model directory. |
| `--input_tsv` | yes | none | TSV file with `text` and reference audio columns. |
| `--output_dir` | yes | none | Directory for `wav/`, metadata TSV, and failure TSV. |
| `--output_tsv_name` | no | `generated.tsv` | Metadata TSV filename inside `output_dir`. |
| `--failed_tsv_name` | no | `failed.tsv` | Failure TSV filename inside `output_dir`. |
| `--lang` | no | none | Adds language token to text: `zh`, `en`, `ja`, `yue`, or `ko`. Recommended. |
| `--text_frontend` | no | disabled | Enable CosyVoice text frontend normalization. |
| `--batch_size` | no | `4` | Number of TSV rows processed together. |
| `--llm_batch_size` | no | `batch_size` | Micro-batch size for the LLM stage. |
| `--flow_batch_size` | no | `batch_size` | Micro-batch size for flow and vocoder stage. |
| `--flow_n_timesteps` | no | `10` | Flow diffusion steps. Lower is faster but may reduce quality. |
| `--min_token_text_ratio` | no | `2.0` | Minimum generated speech-token/text-token ratio. |
| `--max_token_text_ratio` | no | `20.0` | Maximum generated speech-token/text-token ratio. |
| `--on_error` | no | `skip` | Use `skip` to continue after row failures, or `raise` to stop. |
| `--save_workers` | no | `4` | Number of threads used to save WAV files. |

## Recommended Launch Template

```bash
cd /home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/inference_scripts/v2

python3 cosyvoice_generate_from_tsv_batch.py \
  --model_path /home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/pretrained_models/CosyVoice2-0.5B \
  --input_tsv /home/jinyang_wang/data/input.tsv \
  --output_dir /home/jinyang_wang/data/cosyvoice_outputs \
  --output_tsv_name generated.tsv \
  --failed_tsv_name failed.tsv \
  --lang en \
  --batch_size 4 \
  --llm_batch_size 4 \
  --flow_batch_size 4 \
  --flow_n_timesteps 10 \
  --on_error skip \
  --save_workers 4
```

Adjust `--batch_size`, `--llm_batch_size`, and `--flow_batch_size` downward if GPU
memory is limited. Increase them if there is enough GPU memory and you want better
throughput.

## Runtime Logs

During inference the program prints per-batch timing, audio duration, and RTF
statistics, then prints a final summary:

```text
Input rows: ...
Generated utterances: ...
Failed rows: ...
Overall timing: ...
WAV directory: ...
Metadata TSV: ...
Failure TSV: ...
```
