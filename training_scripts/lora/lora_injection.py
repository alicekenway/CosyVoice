import logging

import torch.nn as nn
from peft import LoraConfig as PeftLoraConfig, get_peft_model

from training_scripts.lora.lora_layer import LoRALinear
from training_scripts.lora.lora_config import LoRAConfig


def _count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def inject_lora_llm(llm_model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Inject LoRA adapters into the Qwen2 backbone inside the CosyVoice LLM.

    Args:
        llm_model: a Qwen2LM or CosyVoice3LM instance
        config: LoRA configuration

    Returns:
        the same model, modified in-place
    """
    # llm_model.llm is Qwen2Encoder, llm_model.llm.model is Qwen2ForCausalLM
    qwen2_model = llm_model.llm.model

    # apply PEFT LoRA to the HuggingFace model
    peft_config = PeftLoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.llm_target_modules,
        bias='none',
        task_type=None,  # not a standard HF task
    )
    llm_model.llm.model = get_peft_model(qwen2_model, peft_config)

    # freeze everything first
    for p in llm_model.parameters():
        p.requires_grad = False
    # unfreeze PEFT adapter params
    for name, p in llm_model.llm.model.named_parameters():
        if 'lora_' in name:
            p.requires_grad = True

    # optionally unfreeze auxiliary layers outside Qwen2
    if config.train_llm_decoder:
        for p in llm_model.llm_decoder.parameters():
            p.requires_grad = True
    if config.train_speech_embedding:
        for p in llm_model.speech_embedding.parameters():
            p.requires_grad = True

    total = _count_params(llm_model, trainable_only=False)
    trainable = _count_params(llm_model, trainable_only=True)
    logging.info(f'LLM LoRA injected: {trainable:,} trainable / {total:,} total params '
                 f'({100 * trainable / total:.2f}%)')
    return llm_model


def inject_lora_flow(flow_model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Inject custom LoRA adapters into the DiT blocks of the Flow model.

    Args:
        flow_model: a CausalMaskedDiffWithDiT instance
        config: LoRA configuration

    Returns:
        the same model, modified in-place
    """
    dit = flow_model.decoder.estimator

    for block_idx, block in enumerate(dit.transformer_blocks):
        if config.dit_target_attn:
            # attention projections
            block.attn.to_q = LoRALinear(block.attn.to_q, config.rank, config.alpha, config.dropout)
            block.attn.to_k = LoRALinear(block.attn.to_k, config.rank, config.alpha, config.dropout)
            block.attn.to_v = LoRALinear(block.attn.to_v, config.rank, config.alpha, config.dropout)
            # to_out is nn.ModuleList([nn.Linear, nn.Dropout]), target index 0
            block.attn.to_out[0] = LoRALinear(block.attn.to_out[0], config.rank, config.alpha, config.dropout)

        if config.dit_target_ff:
            # ff.ff is nn.Sequential(nn.Sequential(Linear, GELU), Dropout, Linear)
            ff_seq = block.ff.ff
            ff_seq[0][0] = LoRALinear(ff_seq[0][0], config.rank, config.alpha, config.dropout)
            ff_seq[2] = LoRALinear(ff_seq[2], config.rank, config.alpha, config.dropout)

    # freeze all non-LoRA params
    for name, p in flow_model.named_parameters():
        if 'lora_' not in name:
            p.requires_grad = False

    total = _count_params(flow_model, trainable_only=False)
    trainable = _count_params(flow_model, trainable_only=True)
    logging.info(f'Flow LoRA injected: {trainable:,} trainable / {total:,} total params '
                 f'({100 * trainable / total:.2f}%)')
    return flow_model
