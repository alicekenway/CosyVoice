import json
import logging
import os

import torch
from peft import PeftModel

from training_scripts.lora.lora_config import LoRAConfig
from training_scripts.lora.lora_injection import inject_lora_flow


def save_lora_model(model, save_dir: str, lora_config: LoRAConfig, info: dict = None):
    """Save only LoRA adapter weights and unfrozen auxiliary layers.

    Args:
        model: the full model (may be DDP-wrapped, so access .module)
        save_dir: directory to save into
        lora_config: the LoRA configuration used during training
        info: optional dict with epoch/step info
    """
    rank = int(os.environ.get('RANK', 0))
    if rank != 0:
        return

    m = model.module if hasattr(model, 'module') else model
    os.makedirs(save_dir, exist_ok=True)

    target = lora_config.target

    if target in ('llm', 'both'):
        # save PEFT adapter
        llm_lora_dir = os.path.join(save_dir, 'llm_lora')
        m.llm.model.save_pretrained(llm_lora_dir)

        # save auxiliary trainable layers
        aux_state = {}
        if lora_config.train_llm_decoder:
            aux_state['llm_decoder'] = m.llm_decoder.state_dict()
        if lora_config.train_speech_embedding:
            aux_state['speech_embedding'] = m.speech_embedding.state_dict()
        if aux_state:
            torch.save(aux_state, os.path.join(save_dir, 'llm_aux.pt'))

    if target in ('flow', 'both'):
        # save custom LoRA params
        lora_state = {k: v for k, v in m.state_dict().items() if 'lora_' in k}
        torch.save(lora_state, os.path.join(save_dir, 'flow_lora.pt'))

    # save metadata
    meta = {
        'target': target,
        'rank': lora_config.rank,
        'alpha': lora_config.alpha,
        'dropout': lora_config.dropout,
    }
    if target in ('llm', 'both'):
        meta['llm_target_modules'] = lora_config.llm_target_modules
        meta['train_llm_decoder'] = lora_config.train_llm_decoder
        meta['train_speech_embedding'] = lora_config.train_speech_embedding
    if target in ('flow', 'both'):
        meta['dit_target_attn'] = lora_config.dit_target_attn
        meta['dit_target_ff'] = lora_config.dit_target_ff
    if info:
        meta['epoch'] = info.get('epoch')
        meta['step'] = info.get('step')
    with open(os.path.join(save_dir, 'lora_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    logging.info(f'[Rank {rank}] LoRA checkpoint saved to {save_dir}')


def load_lora_llm(llm_model, lora_dir: str):
    """Load PEFT LoRA adapter + auxiliary layers onto an LLM model.

    The model should NOT have LoRA injected yet — this function wraps it via PEFT.
    For inference use only. For resuming training, use resume_lora_llm() instead.
    """
    llm_lora_dir = os.path.join(lora_dir, 'llm_lora')
    qwen2_model = llm_model.llm.model
    llm_model.llm.model = PeftModel.from_pretrained(qwen2_model, llm_lora_dir)
    logging.info(f'LLM LoRA adapter loaded from {llm_lora_dir}')

    _load_llm_aux(llm_model, lora_dir)
    return llm_model


def resume_lora_llm(llm_model, lora_dir: str):
    """Load adapter weights into an already-PEFT-wrapped LLM model.

    Call this after inject_lora_llm() to resume training from a checkpoint.
    Unlike load_lora_llm(), this does NOT re-wrap with PeftModel.
    """
    from peft import set_peft_model_state_dict

    llm_lora_dir = os.path.join(lora_dir, 'llm_lora')
    safetensors_path = os.path.join(llm_lora_dir, 'adapter_model.safetensors')
    bin_path = os.path.join(llm_lora_dir, 'adapter_model.bin')
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        adapter_state = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        adapter_state = torch.load(bin_path, map_location='cpu')
    else:
        raise FileNotFoundError(f'No adapter weights found in {llm_lora_dir}')

    set_peft_model_state_dict(llm_model.llm.model, adapter_state)
    logging.info(f'LLM LoRA adapter weights resumed from {llm_lora_dir}')

    _load_llm_aux(llm_model, lora_dir)
    return llm_model


def resume_lora_flow(flow_model, lora_dir: str):
    """Load LoRA weights into an already-injected Flow model.

    Call this after inject_lora_flow() to resume training from a checkpoint.
    """
    lora_path = os.path.join(lora_dir, 'flow_lora.pt')
    lora_state = torch.load(lora_path, map_location='cpu')
    missing, unexpected = flow_model.load_state_dict(lora_state, strict=False)
    logging.info(f'Flow LoRA weights resumed from {lora_dir} '
                 f'(loaded {len(lora_state)} params, {len(missing)} missing)')
    return flow_model


def load_lora_meta(lora_dir: str) -> dict:
    """Load lora_meta.json from a checkpoint directory."""
    meta_path = os.path.join(lora_dir, 'lora_meta.json')
    with open(meta_path) as f:
        return json.load(f)


def _load_llm_aux(llm_model, lora_dir: str):
    """Shared helper: load auxiliary trainable layers (llm_decoder, speech_embedding)."""
    aux_path = os.path.join(lora_dir, 'llm_aux.pt')
    if os.path.exists(aux_path):
        aux_state = torch.load(aux_path, map_location='cpu')
        if 'llm_decoder' in aux_state:
            llm_model.llm_decoder.load_state_dict(aux_state['llm_decoder'])
        if 'speech_embedding' in aux_state:
            llm_model.speech_embedding.load_state_dict(aux_state['speech_embedding'])
        logging.info(f'LLM auxiliary layers loaded from {aux_path}')


def load_lora_flow(flow_model, lora_dir: str):
    """Load custom LoRA weights onto a Flow model.

    This injects LoRA structure first (using config from lora_meta.json), then loads weights.
    """
    meta_path = os.path.join(lora_dir, 'lora_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    config = LoRAConfig(
        target='flow',
        rank=meta['rank'],
        alpha=meta['alpha'],
        dropout=meta.get('dropout', 0.0),
        dit_target_attn=meta.get('dit_target_attn', True),
        dit_target_ff=meta.get('dit_target_ff', False),
    )
    flow_model = inject_lora_flow(flow_model, config)

    lora_state = torch.load(os.path.join(lora_dir, 'flow_lora.pt'), map_location='cpu')
    # the saved keys have the full path; load with strict=False since base params won't match
    missing, unexpected = flow_model.load_state_dict(lora_state, strict=False)
    logging.info(f'Flow LoRA loaded from {lora_dir} '
                 f'(loaded {len(lora_state)} params, {len(missing)} missing, {len(unexpected)} unexpected)')
    return flow_model
