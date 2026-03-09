# CosyVoice v3 Batch Inference

This directory now supports two LLM backends for stage-1 token generation:

- `native`: existing PyTorch path (default)
- `trtllm`: TensorRT-LLM engine path

## 1) Convert HF checkpoint to TensorRT engine

Reuse the runtime conversion toolchain from `runtime/triton_trtllm`:

```bash
python3 CosyVoice/inference_scripts/v3/convert_hf_to_tensorrt.py \
  --hf_model_dir /path/to/cosyvoice3_hf_llm \
  --output_root /path/to/trt_artifacts \
  --dtype bfloat16 \
  --max_batch_size 16 \
  --max_num_tokens 32768
```

Output structure:

- `/path/to/trt_artifacts/trt_weights`
- `/path/to/trt_artifacts/trt_engines`

## 2) Run batch generation with TensorRT-LLM backend

```bash
python3 CosyVoice/inference_scripts/v3/cosyvoice_generate_from_tsv_batch.py \
  --model_path /path/to/CosyVoice3-0.5B \
  --input_tsv /path/to/input.tsv \
  --output_dir /path/to/output \
  --batch_size 8 \
  --llm_backend trtllm \
  --trt_engine_dir /path/to/trt_artifacts/trt_engines \
  --trt_max_input_len 512 \
  --trt_max_output_len 2048 \
  --trt_top_k 50 \
  --trt_top_p 0.95 \
  --trt_temperature 0.8
```

## Notes

- `--trt_engine_dir` is required when `--llm_backend trtllm`.
- `trtllm-build` and Python package `tensorrt_llm` must be installed in the runtime environment.
- The TensorRT backend is intended for CosyVoice tokenizer/engine pairs that generate `<|s_x|>` speech tokens.
