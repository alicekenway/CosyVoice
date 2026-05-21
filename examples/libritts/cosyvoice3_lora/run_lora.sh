#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
# LoRA fine-tuning for CosyVoice3
. ./path.sh || exit 1;

stage=-1
stop_stage=6

pretrained_model_dir=../../../pretrained_models/Fun-CosyVoice3-0.5B
data_dir=/path/to/your/data    # <-- change this to your data directory

# =====================================================
# Stage 0: Prepare data
# Expects $data_dir/{train,dev} with wav.scp, text, utt2spk, spk2utt
# =====================================================
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in train dev; do
    mkdir -p data/$x
    python local/prepare_data.py --src_dir $data_dir/$x --des_dir data/$x
  done
fi

# =====================================================
# Stage 1: Extract speaker embeddings
# =====================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding"
  for x in train dev; do
    tools/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

# =====================================================
# Stage 2: Extract speech tokens
# =====================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech tokens"
  for x in train dev; do
    tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v3.onnx
  done
fi

# =====================================================
# Stage 3: Make parquet data
# =====================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare parquet format data"
  for x in train dev; do
    mkdir -p data/$x/parquet
    ../../../tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi

# =====================================================
# Training settings
# =====================================================
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

# =====================================================
# Stage 4: LoRA train LLM
# =====================================================
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "LoRA fine-tune LLM"
  cat data/train/parquet/data.list > data/train.data.list
  cat data/dev/parquet/data.list > data/dev.data.list
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    ../../../training_scripts/bin/train_lora.py \
    --train_engine $train_engine \
    --config conf/cosyvoice3_lora_llm.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
    --onnx_path $pretrained_model_dir \
    --model llm \
    --checkpoint $pretrained_model_dir/llm.pt \
    --model_dir `pwd`/exp/lora_llm/$train_engine \
    --tensorboard_dir `pwd`/tensorboard/lora_llm/$train_engine \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    --use_amp
fi

# =====================================================
# Stage 5: LoRA train Flow
# =====================================================
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "LoRA fine-tune Flow"
  cat data/train/parquet/data.list > data/train.data.list
  cat data/dev/parquet/data.list > data/dev.data.list
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    ../../../training_scripts/bin/train_lora.py \
    --train_engine $train_engine \
    --config conf/cosyvoice3_lora_flow.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
    --onnx_path $pretrained_model_dir \
    --model flow \
    --checkpoint $pretrained_model_dir/flow.pt \
    --model_dir `pwd`/exp/lora_flow/$train_engine \
    --tensorboard_dir `pwd`/tensorboard/lora_flow/$train_engine \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    --use_amp
fi

# =====================================================
# Stage 6: Inference with LoRA
# =====================================================
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Run inference with LoRA adapters"
  # pick the best checkpoint (or use epoch_X_whole)
  llm_ckpt=`pwd`/exp/lora_llm/$train_engine/epoch_49_whole
  flow_ckpt=`pwd`/exp/lora_flow/$train_engine/epoch_99_whole
  python ../../../training_scripts/inference_lora.py \
    --model_dir $pretrained_model_dir \
    --llm_lora_path $llm_ckpt \
    --flow_lora_path $flow_ckpt \
    --text "Hello, this is a test." \
    --output_dir `pwd`/output
fi
