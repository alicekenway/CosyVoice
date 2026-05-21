#!/usr/bin/env python3
"""Inference with LoRA adapters on CosyVoice3.

Usage:
    python training_scripts/inference_lora.py \
        --model_dir pretrained_models/Fun-CosyVoice3-0.5B \
        --llm_lora_path exp/lora_llm/torch_ddp/epoch_10_whole \
        --text "Hello, this is a test."
"""
import argparse
import logging
import os

import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice3
from training_scripts.save_load import load_lora_llm, load_lora_flow


def get_args():
    parser = argparse.ArgumentParser(description='CosyVoice3 inference with LoRA')
    parser.add_argument('--model_dir', required=True, help='base model dir')
    parser.add_argument('--llm_lora_path', default=None, help='LLM LoRA checkpoint dir')
    parser.add_argument('--flow_lora_path', default=None, help='Flow LoRA checkpoint dir')
    parser.add_argument('--text', required=True, help='text to synthesize')
    parser.add_argument('--prompt_text', default=None, help='prompt text for zero-shot')
    parser.add_argument('--prompt_wav', default=None, help='prompt wav path for zero-shot')
    parser.add_argument('--output_dir', default='output', help='output directory')
    parser.add_argument('--spk_id', default=None, help='speaker id for SFT mode')
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # load base model
    logging.info(f'Loading base model from {args.model_dir}')
    cosyvoice = CosyVoice3(args.model_dir)

    # load LoRA adapters
    if args.llm_lora_path:
        logging.info(f'Loading LLM LoRA from {args.llm_lora_path}')
        load_lora_llm(cosyvoice.model.llm, args.llm_lora_path)

    if args.flow_lora_path:
        logging.info(f'Loading Flow LoRA from {args.flow_lora_path}')
        load_lora_flow(cosyvoice.model.flow, args.flow_lora_path)

    # run inference
    os.makedirs(args.output_dir, exist_ok=True)

    if args.prompt_wav and args.prompt_text:
        # zero-shot mode
        prompt_speech_16k = torchaudio.load(args.prompt_wav)[0]
        for i, result in enumerate(cosyvoice.inference_zero_shot(
                args.text, args.prompt_text, prompt_speech_16k)):
            out_path = os.path.join(args.output_dir, f'zero_shot_{i}.wav')
            torchaudio.save(out_path, result['tts_speech'], cosyvoice.sample_rate)
            logging.info(f'Saved {out_path}')
    elif args.spk_id:
        # SFT mode
        for i, result in enumerate(cosyvoice.inference_sft(args.text, args.spk_id)):
            out_path = os.path.join(args.output_dir, f'sft_{i}.wav')
            torchaudio.save(out_path, result['tts_speech'], cosyvoice.sample_rate)
            logging.info(f'Saved {out_path}')
    else:
        # cross-lingual or basic mode — use instruct if available
        logging.warning('No prompt_wav/prompt_text or spk_id provided. '
                        'Using cross_lingual mode requires prompt_wav.')
        logging.info('Please provide --prompt_wav and --prompt_text for zero-shot, '
                     'or --spk_id for SFT mode.')


if __name__ == '__main__':
    main()
