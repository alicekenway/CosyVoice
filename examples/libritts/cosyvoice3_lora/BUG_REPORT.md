# CosyVoice3 LoRA Add-on Bug Report

Date: 2026-04-16

Scope reviewed:
- `training_scripts/`
- `examples/libritts/cosyvoice3_lora/`
- Relevant runtime and training paths in the main `cosyvoice/` project

Review goal:
- Check whether the `cosyvoice3_lora` add-on is functionally correct when used against the current project, not just whether the new files are locally well-formed.

Validation performed:
- Static code review across the add-on and the existing training/inference stack
- Cross-check against the working `examples/libritts/cosyvoice3/` example
- Import-level and syntax-level validation where possible
- `python -m py_compile` passed for the added Python files

What this report is:
- A handoff-ready breakdown of the confirmed or highly likely bugs
- For each bug: symptom, impact, evidence, a concrete demonstration path, and suggested ownership

What this report is not:
- A full runtime training validation on GPUs
- A claim that these are the only issues in the add-on

## Executive Summary

I found 5 meaningful issues:

1. The example data-preparation stage is wired incorrectly for CosyVoice3 LLM training and will fail before LLM LoRA training can run.
2. The provided stage-6 inference example cannot generate any output as written.
3. Zero-shot inference in `inference_lora.py` passes the wrong type into the runtime API and will fail if the user does provide prompt inputs.
4. `--lora_checkpoint` is presented as training resume, but it does not restore optimizer or scheduler state, so it is only a partial resume.
5. Flow-only LoRA is still hard-coupled to `peft`, and `peft` is not declared in project requirements, so a clean environment can fail before flow training even starts.

Recommended triage priority:

- P0: Bug 1, Bug 2, Bug 3
- P1: Bug 5
- P2: Bug 4

---

## Bug 1: CosyVoice3 LLM LoRA data preparation is incompatible with the main CosyVoice3 training path

Severity: P0

### Short description

The example script in `examples/libritts/cosyvoice3_lora/run_lora.sh` prepares data in a way that is inconsistent with CosyVoice3 LLM training requirements.

There are two separate problems here:

1. It calls a non-existent relative path:
   - `python local/prepare_data.py ...`
2. It does not provide `--instruct`, even though `CosyVoice3LM.forward()` requires `instruct_token` in the batch.

Either one is enough to break the example path. Together, they make stage 0 plus LLM training unusable as documented.

### Evidence

#### A. Broken `local/prepare_data.py` path in the LoRA example

File:
- [run_lora.sh](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/examples/libritts/cosyvoice3_lora/run_lora.sh:20)

Current line:

```bash
python local/prepare_data.py --src_dir $data_dir/$x --des_dir data/$x
```

But the reviewed tree for `examples/libritts/cosyvoice3_lora/` does not contain a `local/` directory.

The working non-LoRA CosyVoice3 example instead uses:
- [examples/libritts/cosyvoice3/run.sh](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/examples/libritts/cosyvoice3/run.sh:24)

```bash
python local/prepare_data.py --src_dir $data_dir/LibriTTS/$x --des_dir data/$x --instruct "You are a helpful assistant.<|endofprompt|>"
```

That working script lives in a sibling directory that does have the expected local helper path layout.

#### B. Missing `instruct` data for CosyVoice3 LLM training

`CosyVoice3LM.forward()` unconditionally reads `instruct_token`:
- [cosyvoice/llm/llm.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/cosyvoice/llm/llm.py:387)

Relevant code:

```python
if self.__class__.__name__ == 'CosyVoice3LM':
    instruct_token = batch['instruct_token'].to(device)
    instruct_token_len = batch['instruct_token_len'].to(device)
```

The dataset only creates `instruct_token` if the sample has `instruct`:
- [cosyvoice/dataset/processor.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/cosyvoice/dataset/processor.py:263)
- [cosyvoice/dataset/processor.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/cosyvoice/dataset/processor.py:403)

And `prepare_data.py` only writes an `instruct` file when `--instruct` is passed:
- [prepare_data.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/examples/libritts/cosyvoice/local/prepare_data.py:43)

The LoRA example does not pass `--instruct`:
- [run_lora.sh](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/examples/libritts/cosyvoice3_lora/run_lora.sh:20)

### Expected behavior

The example should prepare a dataset that is valid for `CosyVoice3LM` training, including the required instruction sequence.

### Actual behavior

Two likely failure modes:

1. Stage 0 fails immediately because `local/prepare_data.py` is not found from the `cosyvoice3_lora` directory.
2. If the helper path is fixed manually but `--instruct` remains missing, LLM LoRA training will fail on the first forward pass with a missing `instruct_token` key in the batch.

### Demonstration

#### Demonstration A: Path failure

From `examples/libritts/cosyvoice3_lora/`, stage 0 attempts:

```bash
python local/prepare_data.py --src_dir $data_dir/train --des_dir data/train
```

But the reviewed add-on directory does not ship `examples/libritts/cosyvoice3_lora/local/prepare_data.py`.

Result:
- The documented stage-0 command path is invalid as checked into the repo.

#### Demonstration B: Missing `instruct_token`

Control flow:

1. `run_lora.sh` stage 0 writes `wav.scp/text/utt2spk/spk2utt`
2. It does not write `instruct`
3. `make_parquet_list.py` therefore produces samples without `instruct`
4. `tokenize()` cannot create `instruct_token`
5. `padding()` only includes `instruct_token` if all samples have it
6. `CosyVoice3LM.forward()` reads `batch['instruct_token']` unconditionally

This leads to a runtime key error during LLM training.

### Root cause

The LoRA example was copied or adapted without carrying over the CosyVoice3-specific instruction-data requirement and without preserving the helper path layout used by the main example.

### Suggested owner

- Example / recipe owner
- Secondary review from the LLM training owner

### Suggested fix

1. Fix the helper script path in `run_lora.sh`
2. Pass an `--instruct` string, matching the base CosyVoice3 recipe
3. Add a note in the LoRA example that `CosyVoice3LM` training requires instruction data
4. Optionally add an explicit preflight check in `train_lora.py` for `CosyVoice3LM` batches missing `instruct_token`

---

## Bug 2: The stage-6 LoRA inference example cannot synthesize audio as written

Severity: P0

### Short description

The example inference command in stage 6 does not pass enough information for any synthesis branch to execute.

### Evidence

The stage-6 command is:
- [run_lora.sh](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/examples/libritts/cosyvoice3_lora/run_lora.sh:133)

```bash
python ../../../training_scripts/inference_lora.py \
  --model_dir $pretrained_model_dir \
  --llm_lora_path $llm_ckpt \
  --flow_lora_path $flow_ckpt \
  --text "Hello, this is a test." \
  --output_dir `pwd`/output
```

But `inference_lora.py` only synthesizes in two branches:
- zero-shot: requires both `--prompt_wav` and `--prompt_text`
- SFT: requires `--spk_id`

See:
- [training_scripts/inference_lora.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/inference_lora.py:54)
- [training_scripts/inference_lora.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/inference_lora.py:62)

The fallback branch only logs and exits:
- [training_scripts/inference_lora.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/inference_lora.py:68)

### Expected behavior

The example command in the recipe should produce at least one waveform file or clearly demonstrate a valid inference mode.

### Actual behavior

It prints a warning and does not write output audio.

### Demonstration

If the user runs the documented stage-6 command exactly as written, control flow is:

1. `args.prompt_wav` is `None`
2. `args.prompt_text` is `None`
3. `args.spk_id` is `None`
4. The script enters the final `else` branch
5. No synthesis method is called
6. No `.wav` file is saved

### Root cause

The example command and the script interface are out of sync. The example looks like a minimal smoke test, but the script is implemented only for zero-shot or SFT paths.

### Suggested owner

- Example / recipe owner
- Inference utility owner

### Suggested fix

Choose one:

1. Update stage 6 to pass a valid zero-shot example with `--prompt_wav` and `--prompt_text`
2. Update stage 6 to use `--spk_id` and document required speaker setup
3. Extend `inference_lora.py` to support a default path if the project intends one

---

## Bug 3: `inference_lora.py` passes the wrong type into `CosyVoice3.inference_zero_shot()`

Severity: P0

### Short description

Even if the user fixes Bug 2 by providing prompt inputs, zero-shot inference is still broken because the script loads the prompt wav into a tensor and passes that tensor where the runtime expects a wav path-like input.

### Evidence

The LoRA inference utility does:
- [training_scripts/inference_lora.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/inference_lora.py:56)

```python
prompt_speech_16k = torchaudio.load(args.prompt_wav)[0]
for i, result in enumerate(cosyvoice.inference_zero_shot(
        args.text, args.prompt_text, prompt_speech_16k)):
```

But the runtime API ultimately expects `prompt_wav` to be used as an input to frontend extraction helpers that call `load_wav(prompt_wav, ...)`:

- [cosyvoice/cli/frontend.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/cosyvoice/cli/frontend.py:168)
- [cosyvoice/utils/file_utils.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/cosyvoice/utils/file_utils.py:44)

`load_wav()` does:

```python
speech, sample_rate = torchaudio.load(wav, backend='soundfile')
```

That is a file-loading path, not a tensor input path.

### Expected behavior

`inference_lora.py` should pass `args.prompt_wav` through unchanged if it is using the `CosyVoice3.inference_zero_shot()` API.

### Actual behavior

It converts the path into a tensor too early, then passes the tensor into a code path that tries to treat it as a wav input source.

### Demonstration

Assume the user runs:

```bash
python training_scripts/inference_lora.py \
  --model_dir ... \
  --llm_lora_path ... \
  --flow_lora_path ... \
  --text "hello" \
  --prompt_text "prompt" \
  --prompt_wav prompt.wav
```

Control flow:

1. `torchaudio.load(args.prompt_wav)[0]` creates a waveform tensor
2. That tensor is passed into `CosyVoice3.inference_zero_shot(...)`
3. `CosyVoice3.inference_zero_shot()` forwards it to frontend processing
4. Frontend eventually calls `load_wav(prompt_wav, ...)`
5. `load_wav()` expects a loadable wav input, not the already-loaded tensor object used here

Result:
- Zero-shot inference fails despite valid CLI arguments

### Root cause

The utility script mixed two different API styles:

- Low-level style: pass preloaded tensors directly
- High-level `CosyVoice3` style: pass raw user inputs and let frontend preprocessing handle them

The script uses the high-level object but applies low-level preprocessing on the prompt wav.

### Suggested owner

- Inference utility owner
- Secondary review from the `CosyVoice3` runtime owner

### Suggested fix

Replace:

```python
prompt_speech_16k = torchaudio.load(args.prompt_wav)[0]
cosyvoice.inference_zero_shot(args.text, args.prompt_text, prompt_speech_16k)
```

With:

```python
cosyvoice.inference_zero_shot(args.text, args.prompt_text, args.prompt_wav)
```

Also remove the now-unused `torchaudio.load(...)` preprocessing step in that branch.

---

## Bug 4: `--lora_checkpoint` is only a partial resume, not a real optimizer/scheduler resume

Severity: P2

### Short description

The training script presents `--lora_checkpoint` as resume support, but it only restores adapter weights and metadata. It does not restore optimizer state, scheduler state, gradient-scaler state, or DeepSpeed training state.

This is not an immediate crash bug, but it is a correctness and experiment-reproducibility bug.

### Evidence

Resume path:
- [training_scripts/bin/train_lora.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/bin/train_lora.py:205)

Saved data:
- [training_scripts/save_load.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/save_load.py:12)

What is actually saved:

- For LLM:
  - PEFT adapter files
  - Optional `llm_decoder`
  - Optional `speech_embedding`
  - `lora_meta.json` with `epoch` and `step`
- For Flow:
  - `flow_lora.pt`
  - `lora_meta.json`

What is not saved:

- Optimizer state
- Scheduler state
- AMP scaler state
- DeepSpeed engine state

Then on restart:
- [training_scripts/bin/train_lora.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/bin/train_lora.py:225)

The script builds a fresh optimizer and only sets scheduler step numerically:

```python
model, optimizer, scheduler = init_lora_optimizer_and_scheduler(args, configs, model)
scheduler.set_step(start_step)
```

That is not equivalent to restoring optimizer internals like Adam moments.

### Expected behavior

If the flag is described as training resume, it should restore training state in a way that is materially equivalent to continuing from the same run.

### Actual behavior

The model weights resume, but optimizer dynamics reset.

### Demonstration

Practical consequence:

1. Train to step N
2. Save LoRA checkpoint
3. Restart with `--lora_checkpoint`
4. Model weights continue from step N
5. AdamW momentum and variance buffers restart from zero
6. The resumed run is no longer training-equivalent to a continuous run

This is especially relevant for:

- small LoRA datasets
- short training schedules
- LR-sensitive tuning
- any use of DeepSpeed where users may assume full engine resume

### Root cause

The implementation is centered on lightweight adapter export rather than full training checkpointing, but the CLI and code comments imply a stronger resume guarantee than the implementation provides.

### Suggested owner

- Training infrastructure owner

### Suggested fix

Choose one:

1. Rename and document it as weight-only resume
2. Implement a true training-state resume path
3. Separate the concepts clearly:
   - `--lora_init_from`
   - `--lora_resume_training`

---

## Bug 5: Flow-only LoRA still depends on `peft`, and `peft` is not declared in repo requirements

Severity: P1

### Short description

The Flow LoRA path uses custom `LoRALinear` modules and should not logically require PEFT, but current imports make `peft` mandatory even when the user only wants Flow LoRA. On top of that, `peft` is not declared in the project requirements files.

### Evidence

Unconditional imports:

- [training_scripts/lora/lora_injection.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/lora/lora_injection.py:4)
- [training_scripts/save_load.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/save_load.py:6)
- [training_scripts/bin/train_lora.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/bin/train_lora.py:36)

These are imported before the script knows whether `--model llm` or `--model flow` is requested.

The repo requirements files do not declare `peft` or `safetensors`:

- [requirements.txt](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/requirements.txt:1)
- [requirements.cpu.txt](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/requirements.cpu.txt:1)

In the review environment, importing `peft` failed with:

```text
ModuleNotFoundError: No module named 'peft'
```

### Expected behavior

At minimum:

- the project requirements should declare dependencies needed by the add-on

Preferably:

- Flow LoRA should be runnable without `peft` if the implementation does not actually use it

### Actual behavior

A clean environment that satisfies the repo requirements can still fail before any Flow LoRA logic starts.

### Demonstration

User intent:

```bash
python training_scripts/bin/train_lora.py --model flow ...
```

Import chain:

1. `train_lora.py` imports `training_scripts.lora.lora_injection`
2. `lora_injection.py` imports `peft`
3. Import fails before flow-specific code runs

This means:
- flow-only LoRA is blocked by an undeclared dependency that it may not need

### Root cause

The add-on LLM and Flow code paths were coupled too early at import time.

### Suggested owner

- Add-on / packaging owner
- Secondary review from whoever owns environment setup documentation

### Suggested fix

1. Add `peft` and `safetensors` to requirements if they are intended hard dependencies
2. If Flow LoRA is meant to be independent:
   - move `peft` imports into LLM-only functions
   - keep Flow-only code importable without PEFT

---

## Suggested Allocation

### Task A: Fix the LoRA example recipe

Owner profile:
- Example / recipe maintainer

Files:
- [examples/libritts/cosyvoice3_lora/run_lora.sh](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/examples/libritts/cosyvoice3_lora/run_lora.sh:1)

Work:
- Fix helper path
- Add `--instruct`
- Make stage-6 command valid
- Add comments explaining valid inference modes

### Task B: Fix `inference_lora.py`

Owner profile:
- Runtime / inference utility maintainer

Files:
- [training_scripts/inference_lora.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/inference_lora.py:1)

Work:
- Pass `args.prompt_wav` directly
- Decide whether to support a default non-prompt mode
- Improve error messages so invalid invocation fails fast

### Task C: Decide resume semantics

Owner profile:
- Training infrastructure maintainer

Files:
- [training_scripts/bin/train_lora.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/bin/train_lora.py:1)
- [training_scripts/save_load.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/save_load.py:1)

Work:
- Either implement full-state resume
- Or explicitly downgrade the semantics in docs and CLI help

### Task D: Fix dependency packaging and import boundaries

Owner profile:
- Packaging / environment maintainer

Files:
- [requirements.txt](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/requirements.txt:1)
- [requirements.cpu.txt](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/requirements.cpu.txt:1)
- [training_scripts/lora/lora_injection.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/lora/lora_injection.py:1)
- [training_scripts/save_load.py](/home/jinyang_wang/Dev/TTS/TTS_cosyvoice/CosyVoice/training_scripts/save_load.py:1)

Work:
- Declare the dependencies or make PEFT lazy-imported and LLM-only

---

## Recommended Fix Order

1. Fix Bug 1
2. Fix Bug 2
3. Fix Bug 3
4. Fix Bug 5
5. Clarify or rework Bug 4

Reasoning:

- Bugs 1 to 3 block the documented happy path
- Bug 5 blocks clean-environment adoption
- Bug 4 affects correctness and experiment continuity, but not first-run usability

---

## Confidence Notes

High confidence:
- Bug 1
- Bug 2
- Bug 3
- Bug 5

Medium-high confidence:
- Bug 4

Bug 4 is not speculative, but its severity depends on how the team intends to define “resume.” If the intended meaning is only “reload adapter weights and continue approximately,” then it is a documentation bug more than a code bug. If the intended meaning is true training continuation, then it is a functional bug.

---

## Minimal Reproduction Checklist

Use this list when assigning or verifying fixes.

### Repro 1: LLM data path

Expected after fix:
- Stage 0 runs from `examples/libritts/cosyvoice3_lora/`
- Generated dataset includes `instruct`
- LLM LoRA training starts without missing `instruct_token`

### Repro 2: Example inference path

Expected after fix:
- Stage 6 example command actually writes wav files

### Repro 3: Zero-shot prompt handling

Expected after fix:
- `inference_lora.py` works when called with:

```bash
python training_scripts/inference_lora.py \
  --model_dir <base_model_dir> \
  --llm_lora_path <llm_ckpt_dir> \
  --flow_lora_path <flow_ckpt_dir> \
  --text "Hello" \
  --prompt_text "Prompt text" \
  --prompt_wav <prompt.wav>
```

### Repro 4: Flow-only environment

Expected after fix:
- A clean install from repo requirements can at least import and start the Flow LoRA path

### Repro 5: Resume semantics

Expected after fix:
- Either full optimizer/scheduler resume works
- Or CLI/docs explicitly say it is weight-only resume

