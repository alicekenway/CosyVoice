#!/usr/bin/env python3
# Copyright 2024 Alibaba Inc (authors: Xiang Lyu)
# LoRA fine-tuning wrapper by training_scripts addon
#
# Licensed under the Apache License, Version 2.0

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.optim as optim
import torch.distributed as dist
import deepspeed

from hyperpyyaml import load_hyperpyyaml
from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_summarywriter,
    wrap_cuda_model,
    check_modify_and_save_config,
    log_per_step, log_per_save,
    batch_forward, batch_backward,
    update_parameter_and_lr,
    cosyvoice_join,
)
from cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR

from training_scripts.lora.lora_config import LoRAConfig
from training_scripts.lora.lora_injection import inject_lora_llm, inject_lora_flow
from training_scripts.save_load import save_lora_model, resume_lora_llm, resume_lora_flow, load_lora_meta


def get_args():
    parser = argparse.ArgumentParser(description='LoRA fine-tuning for CosyVoice')
    parser.add_argument('--train_engine', default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'])
    parser.add_argument('--model', required=True, choices=['llm', 'flow'],
                        help='which component to train')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False)
    parser.add_argument('--onnx_path', required=False)
    parser.add_argument('--checkpoint', help='base model checkpoint')
    parser.add_argument('--lora_checkpoint', default=None,
                        help='resume LoRA training from this checkpoint')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir', default='tensorboard')
    parser.add_argument('--ddp.dist_backend', dest='dist_backend',
                        default='nccl', choices=['nccl', 'gloo'])
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--prefetch', default=100, type=int)
    parser.add_argument('--pin_memory', action='store_true', default=False)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--timeout', default=60, type=int)
    parser.add_argument('--deepspeed.save_states', dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'])
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def init_lora_optimizer_and_scheduler(args, configs, model):
    """Like init_optimizer_and_scheduler but only passes trainable params."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f'Optimizer receives {len(trainable_params)} trainable parameter groups')

    if configs['train_conf']['optim'] == 'adam':
        optimizer = optim.Adam(trainable_params, **configs['train_conf']['optim_conf'])
    elif configs['train_conf']['optim'] == 'adamw':
        optimizer = optim.AdamW(trainable_params, **configs['train_conf']['optim_conf'])
    else:
        raise ValueError("unknown optimizer: " + configs['train_conf']['optim'])

    if configs['train_conf']['scheduler'] == 'warmuplr':
        scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
    elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
        scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
    elif configs['train_conf']['scheduler'] == 'constantlr':
        scheduler = ConstantLR(optimizer)
    else:
        raise ValueError("unknown scheduler: " + configs['train_conf']['scheduler'])

    if args.train_engine == 'deepspeed':
        scheduler_type = type(scheduler)

        def scheduler_fn(opt):
            return scheduler_type(opt, **configs['train_conf'].get('scheduler_conf', {}))

        model, optimizer, _, scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=None,
            lr_scheduler=scheduler_fn,
            model_parameters=trainable_params,
        )

    return model, optimizer, scheduler


class LoRAExecutor(Executor):
    """Executor subclass that saves only LoRA adapter weights."""

    def __init__(self, lora_config: LoRAConfig, **kwargs):
        super().__init__(**kwargs)
        self.lora_config = lora_config

    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        """Cross validation + save LoRA checkpoint."""
        logging.info('Epoch {} Step {} on_batch_end {} CV rank {}'.format(
            self.epoch, self.step + 1, on_batch_end, self.rank))
        model.eval()
        total_num_utts, total_loss_dict = 0, {}
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            info_dict = batch_forward(model, batch_dict, None, info_dict)

            for k, v in info_dict['loss_dict'].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                total_loss_dict[k].append(v.mean().item() * num_utts)
            log_per_step(None, info_dict)

        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict['loss_dict'] = total_loss_dict
        log_per_save(writer, info_dict)

        # save LoRA adapter instead of full model
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end \
            else 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        save_dir = os.path.join(info_dict['model_dir'], model_name)
        save_lora_model(model, save_dir, self.lora_config,
                        info={'epoch': self.epoch, 'step': self.step})


@record
def main():
    args = get_args()
    os.environ['onnx_path'] = args.onnx_path or ''
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # load config, only keep the model we're training
    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if args.qwen_pretrain_path is not None:
        override_dict['qwen_pretrain_path'] = args.qwen_pretrain_path
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    configs['train_conf'].update(vars(args))

    # parse LoRA config from yaml
    lora_config = LoRAConfig.from_dict(configs.get('lora_conf', {}))
    logging.info(f'LoRA config: {lora_config}')

    # init distributed
    init_distributed(args)

    # get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan=False, dpo=False)

    # save config
    configs = check_modify_and_save_config(args, configs)

    # tensorboard
    writer = init_summarywriter(args)

    # load base model + checkpoint
    model = configs[args.model]
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            logging.info(f'Base model loaded from {args.checkpoint}')
        else:
            logging.warning(f'checkpoint {args.checkpoint} does not exist!')

    # inject LoRA adapters
    if args.model == 'llm' and lora_config.target in ('llm', 'both'):
        model = inject_lora_llm(model, lora_config)
    elif args.model == 'flow' and lora_config.target in ('flow', 'both'):
        model = inject_lora_flow(model, lora_config)
    else:
        raise ValueError(f'--model={args.model} but lora_conf.target={lora_config.target}, '
                         f'they must be compatible')

    # resume from LoRA checkpoint if provided
    if args.lora_checkpoint is not None:
        if os.path.isdir(args.lora_checkpoint):
            meta = load_lora_meta(args.lora_checkpoint)
            if args.model == 'llm':
                model = resume_lora_llm(model, args.lora_checkpoint)
            elif args.model == 'flow':
                model = resume_lora_flow(model, args.lora_checkpoint)
            start_step = meta.get('step', 0)
            start_epoch = meta.get('epoch', -1)
            logging.info(f'Resumed LoRA from {args.lora_checkpoint} '
                         f'(epoch={start_epoch}, step={start_step})')
        else:
            logging.warning(f'lora_checkpoint {args.lora_checkpoint} does not exist, '
                            f'starting from scratch')

    # wrap with DDP / DeepSpeed
    model = wrap_cuda_model(args, model)

    # optimizer with only trainable params
    model, optimizer, scheduler = init_lora_optimizer_and_scheduler(args, configs, model)
    scheduler.set_step(start_step)

    # save init checkpoint
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_dir = os.path.join(info_dict['model_dir'], 'init')
    save_lora_model(model, save_dir, lora_config, info={'epoch': start_epoch, 'step': start_step})

    # executor
    executor = LoRAExecutor(lora_config=lora_config)
    executor.step = start_step

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    logging.info(f'start step {start_step} start epoch {start_epoch}')

    # training loop
    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo",
                                    timeout=datetime.timedelta(seconds=args.timeout))
        executor.train_one_epoc(model, optimizer, scheduler,
                                train_data_loader, cv_data_loader,
                                writer, info_dict, scaler, group_join)
        dist.destroy_process_group(group_join)


if __name__ == '__main__':
    main()
