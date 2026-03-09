from dataclasses import dataclass
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from batch_types import PreparedRow, SegmentInput, SynthesisResult
from frontend_batch import collate_segment_inputs
from io_utils import chunked

@dataclass(frozen=True)
class SegmentTask:
    row_index: int
    segment_index: int
    segment_input: SegmentInput


class StagedBatchInferenceRunner:
    def __init__(
        self,
        cosyvoice,
        llm_batch_size: int,
        flow_batch_size: int,
        on_error: str,
        min_token_text_ratio: float = 2.0,
        max_token_text_ratio: float = 20.0,
        flow_n_timesteps: int = 10,
        llm_backend: str = "native",
        trt_engine_dir: str | None = None,
        trt_max_input_len: int = 512,
        trt_max_output_len: int = 2048,
        trt_kv_cache_free_gpu_memory_fraction: float = 0.6,
        trt_temperature: float = 0.8,
        trt_top_k: int = 50,
        trt_top_p: float = 0.95,
    ):
        self.cosyvoice = cosyvoice
        self.model = cosyvoice.model
        self.on_error = on_error
        self.llm_batch_size = llm_batch_size
        self.flow_batch_size = flow_batch_size
        self.min_token_text_ratio = min_token_text_ratio
        self.max_token_text_ratio = max_token_text_ratio
        self.flow_n_timesteps = flow_n_timesteps
        self.llm_backend = llm_backend
        self.trt_engine_dir = trt_engine_dir
        self.trt_max_input_len = trt_max_input_len
        self.trt_max_output_len = trt_max_output_len
        self.trt_kv_cache_free_gpu_memory_fraction = trt_kv_cache_free_gpu_memory_fraction
        self.trt_temperature = trt_temperature
        self.trt_top_k = trt_top_k
        self.trt_top_p = trt_top_p

        self._trt_runner = None
        self._trt_end_id: int | None = None
        self._speech_token_pattern = re.compile(r"<\|s_(\d+)\|>")
        self._init_llm_backend()

    def _init_llm_backend(self) -> None:
        if self.llm_backend == "native":
            return
        if self.llm_backend != "trtllm":
            raise ValueError(f"Unsupported --llm_backend: {self.llm_backend}")
        if not self.trt_engine_dir:
            raise ValueError("--trt_engine_dir is required when --llm_backend is trtllm")
        if not hasattr(self.cosyvoice.frontend, "tokenizer"):
            raise RuntimeError("CosyVoice frontend tokenizer is required for trtllm backend")
        if not hasattr(self.cosyvoice.frontend.tokenizer, "tokenizer"):
            raise RuntimeError("Expected tokenizer wrapper with .tokenizer for trtllm backend")

        try:
            import tensorrt_llm  # pylint: disable=import-outside-toplevel
            from tensorrt_llm.runtime import ModelRunnerCpp  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "Failed to import tensorrt_llm. Install TensorRT-LLM first for --llm_backend trtllm."
            ) from exc

        runtime_rank = tensorrt_llm.mpi_rank()
        self._trt_runner = ModelRunnerCpp.from_dir(
            engine_dir=self.trt_engine_dir,
            rank=runtime_rank,
            max_output_len=self.trt_max_output_len,
            enable_context_fmha_fp32_acc=False,
            max_batch_size=self.llm_batch_size,
            max_input_len=self.trt_max_input_len,
            kv_cache_free_gpu_memory_fraction=self.trt_kv_cache_free_gpu_memory_fraction,
            cuda_graph_mode=False,
            gather_generation_logits=False,
        )

        hf_tokenizer = self.cosyvoice.frontend.tokenizer.tokenizer
        eos1_token_id = hf_tokenizer.convert_tokens_to_ids("<|eos1|>")
        if eos1_token_id is None or eos1_token_id < 0:
            eos1_token_id = hf_tokenizer.eos_token_id
        if eos1_token_id is None:
            raise RuntimeError("Cannot resolve end token id for trtllm backend")
        self._trt_end_id = int(eos1_token_id)

    @torch.inference_mode()
    def run_batch(self, prepared_rows: Sequence[PreparedRow]) -> List[SynthesisResult]:
        segment_tasks: List[SegmentTask] = []
        for row_index, prepared_row in enumerate(prepared_rows):
            for segment_index, segment_input in enumerate(prepared_row.segment_inputs):
                segment_tasks.append(
                    SegmentTask(
                        row_index=row_index,
                        segment_index=segment_index,
                        segment_input=segment_input,
                    )
                )

        row_errors: Dict[int, str] = {}
        generated_tokens: Dict[Tuple[int, int], List[int]] = {}
        generated_speeches: Dict[Tuple[int, int], torch.Tensor] = {}

        for llm_batch in chunked(segment_tasks, self.llm_batch_size):
            active_tasks = [task for task in llm_batch if task.row_index not in row_errors]
            if not active_tasks:
                continue
            try:
                batched_tokens = self._run_llm_stage_batch(
                    [task.segment_input for task in active_tasks]
                )
            except Exception as exc:  # pylint: disable=broad-except
                error_message = f"LLM stage failed: {exc}"
                if self.on_error == "raise":
                    raise RuntimeError(error_message) from exc
                for task in active_tasks:
                    row_errors[task.row_index] = error_message
                continue

            for task, token_ids in zip(active_tasks, batched_tokens):
                generated_tokens[(task.row_index, task.segment_index)] = token_ids

        for flow_batch in chunked(segment_tasks, self.flow_batch_size):
            active_tasks: List[SegmentTask] = []
            active_tokens: List[List[int]] = []
            for task in flow_batch:
                if task.row_index in row_errors:
                    continue
                token_key = (task.row_index, task.segment_index)
                if token_key not in generated_tokens:
                    error_message = "Missing LLM output tokens for segment"
                    if self.on_error == "raise":
                        raise RuntimeError(error_message)
                    row_errors[task.row_index] = error_message
                    continue
                active_tasks.append(task)
                active_tokens.append(generated_tokens[token_key])

            if not active_tasks:
                continue
            try:
                batched_speeches = self._run_flow_hift_stage_batch(
                    [task.segment_input.model_input for task in active_tasks],
                    active_tokens,
                )
            except Exception as exc:  # pylint: disable=broad-except
                error_message = f"Flow/HiFi stage failed: {exc}"
                if self.on_error == "raise":
                    raise RuntimeError(error_message) from exc
                for task in active_tasks:
                    row_errors[task.row_index] = error_message
                continue

            for task, speech in zip(active_tasks, batched_speeches):
                generated_speeches[(task.row_index, task.segment_index)] = speech

        results: List[SynthesisResult] = []
        for row_index, prepared_row in enumerate(prepared_rows):
            if row_index in row_errors:
                results.append(
                    SynthesisResult(
                        row=prepared_row.row,
                        text_for_metadata=prepared_row.text_for_metadata,
                        error_message=row_errors[row_index],
                    )
                )
                continue

            segment_speeches: List[torch.Tensor] = []
            for segment_index in range(len(prepared_row.segment_inputs)):
                segment_key = (row_index, segment_index)
                if segment_key not in generated_speeches:
                    error_message = "Missing speech output for segment"
                    if self.on_error == "raise":
                        raise RuntimeError(error_message)
                    row_errors[row_index] = error_message
                    break
                segment_speeches.append(generated_speeches[segment_key])
            if row_index in row_errors:
                results.append(
                    SynthesisResult(
                        row=prepared_row.row,
                        text_for_metadata=prepared_row.text_for_metadata,
                        error_message=row_errors[row_index],
                    )
                )
                continue

            results.append(
                SynthesisResult(
                    row=prepared_row.row,
                    text_for_metadata=prepared_row.text_for_metadata,
                    speech=self._concat_segments(segment_speeches),
                )
            )
        return results

    def _run_llm_stage_batch(self, segment_inputs: Sequence[SegmentInput]) -> List[List[int]]:
        if self.llm_backend == "trtllm":
            return self._run_llm_stage_batch_trtllm(segment_inputs)
        return self._run_llm_stage_batch_native(segment_inputs)

    def _run_llm_stage_batch_native(self, segment_inputs: Sequence[SegmentInput]) -> List[List[int]]:
        model_inputs = [item.model_input for item in segment_inputs]
        batch_inputs = collate_segment_inputs(
            [SegmentInput(normalized_text="", model_input=item) for item in model_inputs]
        )

        llm_module = self.model.llm
        if not hasattr(llm_module, "llm") or not hasattr(llm_module, "llm_decoder"):
            raise RuntimeError("Current LLM module does not support strict batch decode")
        if not hasattr(llm_module, "speech_embedding"):
            raise RuntimeError("Current LLM module does not expose speech_embedding")

        text = batch_inputs["text"].to(self.model.device)
        text_len = batch_inputs["text_len"].to(self.model.device)
        batch_size = text.shape[0]

        uses_transformer_path = hasattr(llm_module, "text_embedding") and hasattr(llm_module, "encode")
        uses_qwen_path = (
            hasattr(llm_module.llm, "model")
            and hasattr(llm_module.llm.model, "model")
            and hasattr(llm_module.llm.model.model, "embed_tokens")
        )
        if uses_transformer_path:
            token_table = llm_module.llm_embedding.weight
            text_emb = llm_module.text_embedding(text.to(torch.long))
            text_encoded, text_encoded_len = llm_module.encode(text_emb, text_len)
            llm_embedding = batch_inputs["llm_embedding"].to(self.model.device)
            llm_embedding = F.normalize(llm_embedding, dim=1)
            llm_embedding = llm_module.spk_embed_affine_layer(llm_embedding).unsqueeze(1)
        elif uses_qwen_path:
            token_table = (
                llm_module.llm_embedding.weight
                if hasattr(llm_module, "llm_embedding")
                else llm_module.speech_embedding.weight
            )
            text_encoded = llm_module.llm.model.model.embed_tokens(text.to(torch.long))
            text_encoded_len = text_len.to(torch.long)
            llm_embedding = None
        else:
            raise RuntimeError(
                f"Unsupported strict-batch LLM backend: {llm_module.__class__.__name__}"
            )

        sos_emb = token_table[llm_module.sos].reshape(1, 1, -1)
        task_id_emb = token_table[llm_module.task_id].reshape(1, 1, -1)

        prefix_sequences: List[torch.Tensor] = []
        prefix_lengths: List[int] = []
        for batch_index in range(batch_size):
            effective_len = int(text_encoded_len[batch_index].item())
            prefix_list = [sos_emb.squeeze(0)]
            if llm_embedding is not None:
                prefix_list.append(llm_embedding[batch_index : batch_index + 1].squeeze(0))
            prefix_sequence = torch.cat(
                prefix_list
                + [
                    text_encoded[batch_index : batch_index + 1, :effective_len, :].squeeze(0),
                    task_id_emb.squeeze(0),
                ],
                dim=0,
            )
            prefix_sequences.append(prefix_sequence)
            prefix_lengths.append(prefix_sequence.shape[0])

        min_len = ((text_len.to(torch.float32)) * self.min_token_text_ratio).to(torch.long)
        max_len = ((text_len.to(torch.float32)) * self.max_token_text_ratio).to(torch.long)
        max_decode_steps = int(max_len.max().item())
        max_prefix_len = max(prefix_lengths)
        hidden_dim = int(prefix_sequences[0].shape[1])
        max_sequence_len = max_prefix_len + max_decode_steps

        sequence_buffer = torch.zeros(
            batch_size,
            max_sequence_len,
            hidden_dim,
            device=self.model.device,
            dtype=prefix_sequences[0].dtype,
        )
        for batch_index, prefix_sequence in enumerate(prefix_sequences):
            prefix_len = prefix_lengths[batch_index]
            sequence_buffer[batch_index, :prefix_len] = prefix_sequence
        sequence_len = torch.tensor(prefix_lengths, dtype=torch.long, device=self.model.device)

        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        active_sample_indices = torch.arange(batch_size, device=self.model.device, dtype=torch.long)
        stop_token_ids = getattr(llm_module, "stop_token_ids", [llm_module.eos_token])
        stop_token_set = set(int(token_id) for token_id in stop_token_ids)

        for decode_step in range(max_decode_steps):
            if active_sample_indices.numel() == 0:
                break

            padded_input = sequence_buffer.index_select(0, active_sample_indices)
            padded_len = sequence_len.index_select(0, active_sample_indices)
            active_max_len = int(padded_len.max().item())
            padded_input = padded_input[:, :active_max_len, :]
            y_pred, _ = llm_module.llm(padded_input, padded_len)
            gather_index = (
                (padded_len - 1)
                .view(active_sample_indices.shape[0], 1, 1)
                .expand(-1, 1, y_pred.shape[-1])
            )
            current_last_hidden = y_pred.gather(1, gather_index).squeeze(1)
            logp = llm_module.llm_decoder(current_last_hidden).log_softmax(dim=-1)
            next_active_sample_indices: List[int] = []
            for active_batch_index in range(active_sample_indices.shape[0]):
                sample_index = int(active_sample_indices[active_batch_index].item())
                ignore_eos = decode_step < int(min_len[sample_index].item())
                top_token = llm_module.sampling_ids(
                    logp[active_batch_index],
                    generated_tokens[sample_index],
                    25,
                    ignore_eos=ignore_eos,
                )
                top_token_int = int(top_token.item()) if isinstance(top_token, torch.Tensor) else int(top_token)
                if top_token_int in stop_token_set:
                    continue
                generated_tokens[sample_index].append(top_token_int)
                next_token_embed = llm_module.speech_embedding.weight[top_token_int : top_token_int + 1]
                write_index = int(sequence_len[sample_index].item())
                sequence_buffer[sample_index, write_index : write_index + 1] = next_token_embed
                sequence_len[sample_index] = write_index + 1
                if len(generated_tokens[sample_index]) < int(max_len[sample_index].item()):
                    next_active_sample_indices.append(sample_index)
            active_sample_indices = torch.tensor(
                next_active_sample_indices,
                dtype=torch.long,
                device=self.model.device,
            )

        for token_sequence in generated_tokens:
            if not token_sequence:
                raise RuntimeError("LLM generated zero speech tokens for at least one sample")
        return generated_tokens

    def _run_llm_stage_batch_trtllm(self, segment_inputs: Sequence[SegmentInput]) -> List[List[int]]:
        if self._trt_runner is None or self._trt_end_id is None:
            raise RuntimeError("trtllm backend is not initialized")
        tokenizer_wrapper = self.cosyvoice.frontend.tokenizer

        prompt_template = "<|sos|>{input_text}<|task_id|>"
        batch_input_ids: List[torch.Tensor] = []
        input_lengths: List[int] = []
        for segment_input in segment_inputs:
            llm_prompt = prompt_template.format(input_text=segment_input.normalized_text)
            token_ids = tokenizer_wrapper.encode(llm_prompt)
            input_tensor = torch.tensor(token_ids, dtype=torch.int32)
            batch_input_ids.append(input_tensor)
            input_lengths.append(input_tensor.size(0))

        outputs = self._trt_runner.generate(
            batch_input_ids=batch_input_ids,
            max_new_tokens=self.trt_max_output_len,
            end_id=self._trt_end_id,
            pad_id=self._trt_end_id,
            temperature=self.trt_temperature,
            top_k=self.trt_top_k,
            top_p=self.trt_top_p,
            repetition_penalty=1.1,
            num_return_sequences=1,
            streaming=False,
            output_sequence_lengths=True,
            output_generation_logits=False,
            return_dict=True,
            return_all_generated_tokens=False,
        )

        output_ids = outputs["output_ids"]
        sequence_lengths = outputs["sequence_lengths"]
        batch_size = len(segment_inputs)
        num_output_sents, num_beams, _ = output_ids.size()
        if num_beams != 1:
            raise RuntimeError(f"Expected 1 beam for trtllm backend, got {num_beams}")
        num_return_sequences = num_output_sents // batch_size
        if num_return_sequences != 1:
            raise RuntimeError(
                f"Expected 1 return sequence for trtllm backend, got {num_return_sequences}"
            )

        decoded_token_ids: List[List[int]] = []
        for batch_index in range(batch_size):
            output_end = int(sequence_lengths[batch_index][0].item())
            output_begin = input_lengths[batch_index]
            generated_ids = output_ids[batch_index][0][output_begin:output_end].tolist()
            decoded_text = tokenizer_wrapper.decode(generated_ids)
            speech_tokens = [
                int(match.group(1)) for match in self._speech_token_pattern.finditer(decoded_text)
            ]
            if not speech_tokens:
                raise RuntimeError(
                    "trtllm backend generated no speech tokens; confirm tokenizer/engine compatibility"
                )
            decoded_token_ids.append(speech_tokens)
        return decoded_token_ids

    def _run_flow_hift_stage_batch(
        self,
        model_inputs: Sequence[Dict[str, torch.Tensor]],
        token_ids_list: Sequence[List[int]],
    ) -> List[torch.Tensor]:
        if len(model_inputs) != len(token_ids_list):
            raise ValueError("model_inputs and token_ids_list must have same length")

        batch_inputs = collate_segment_inputs(
            [SegmentInput(normalized_text="", model_input=item) for item in model_inputs]
        )
        flow_module = self.model.flow
        if not all(token_ids_list):
            raise RuntimeError("Token ids list contains empty sequence")

        gen_token_tensors = [
            torch.tensor(token_ids, dtype=torch.int32) for token_ids in token_ids_list
        ]
        gen_token_len = torch.tensor(
            [tensor.shape[0] for tensor in gen_token_tensors],
            dtype=torch.int32,
            device=self.model.device,
        )
        gen_token = pad_sequence(gen_token_tensors, batch_first=True, padding_value=0).to(
            self.model.device
        )

        prompt_token = batch_inputs["flow_prompt_speech_token"].to(self.model.device, dtype=torch.int32)
        prompt_token_len = batch_inputs["flow_prompt_speech_token_len"].to(self.model.device, dtype=torch.int32)
        prompt_feat = batch_inputs["prompt_speech_feat"].to(self.model.device)
        prompt_feat_len = batch_inputs["prompt_speech_feat_len"].to(self.model.device, dtype=torch.int32)
        flow_embedding = batch_inputs["flow_embedding"].to(self.model.device)

        if not hasattr(flow_module, "input_embedding") or not hasattr(flow_module, "decoder"):
            raise RuntimeError("Current flow module does not support strict batch path")
        if not isinstance(flow_module.decoder.estimator, torch.nn.Module):
            raise RuntimeError("TensorRT flow estimator path is unsupported in strict batch mode")

        from cosyvoice.utils.mask import make_pad_mask  # pylint: disable=import-outside-toplevel

        embedding = torch.nn.functional.normalize(flow_embedding, dim=1)
        embedding = flow_module.spk_embed_affine_layer(embedding)

        token_total_tensors: List[torch.Tensor] = []
        for batch_index in range(prompt_token.shape[0]):
            prompt_len = int(prompt_token_len[batch_index].item())
            token_total_tensors.append(
                torch.cat(
                    [
                        prompt_token[batch_index, :prompt_len],
                        gen_token[batch_index, : int(gen_token_len[batch_index].item())],
                    ],
                    dim=0,
                )
            )
        token_total = pad_sequence(token_total_tensors, batch_first=True, padding_value=0)
        token_total_len = torch.tensor(
            [tensor.shape[0] for tensor in token_total_tensors],
            dtype=torch.int32,
            device=self.model.device,
        )
        token_mask = (~make_pad_mask(token_total_len, max_len=token_total.shape[1])).unsqueeze(-1).to(
            embedding
        )
        token_emb = flow_module.input_embedding(torch.clamp(token_total, min=0)) * token_mask

        if hasattr(flow_module, "length_regulator"):
            h, h_lengths = flow_module.encoder(token_emb, token_total_len)
            h = flow_module.encoder_proj(h)
            mel_new_len = torch.tensor(
                [
                    int(length.item() / flow_module.input_frame_rate * 22050 / 256)
                    for length in gen_token_len
                ],
                dtype=torch.int32,
                device=self.model.device,
            )
            mel_total_len = prompt_feat_len + mel_new_len
            h, _ = flow_module.length_regulator(h, mel_total_len)
        else:
            h, _ = flow_module.encoder(token_emb, token_total_len, streaming=False)
            if hasattr(flow_module, "encoder_proj"):
                h = flow_module.encoder_proj(h)
            mel_new_len = gen_token_len * int(flow_module.token_mel_ratio)
            mel_total_len = prompt_feat_len + mel_new_len
            h = h[:, : int(mel_total_len.max().item())]

        max_supported_mel_len = int(h.shape[1])
        mel_total_len = torch.clamp(mel_total_len, max=max_supported_mel_len)
        max_mel_len = int(mel_total_len.max().item())
        h = h[:, :max_mel_len]
        conds = torch.zeros(
            h.shape[0],
            max_mel_len,
            flow_module.output_size,
            device=self.model.device,
            dtype=h.dtype,
        )
        for batch_index in range(h.shape[0]):
            prompt_len = int(prompt_feat_len[batch_index].item())
            conds[batch_index, :prompt_len] = prompt_feat[batch_index, :prompt_len]

        mel_mask = (~make_pad_mask(mel_total_len, max_len=max_mel_len)).unsqueeze(1).to(h)
        mu = h.transpose(1, 2).contiguous()
        cond = conds.transpose(1, 2).contiguous()
        flow_feat = self._batched_cfm_decode(
            decoder=flow_module.decoder,
            mu=mu,
            mask=mel_mask,
            spks=embedding,
            cond=cond,
            streaming=False,
        )

        generated_mels: List[torch.Tensor] = []
        generated_mel_len: List[int] = []
        for batch_index in range(flow_feat.shape[0]):
            start = int(prompt_feat_len[batch_index].item())
            end = int(mel_total_len[batch_index].item())
            sample_mel = flow_feat[batch_index : batch_index + 1, :, start:end]
            generated_mels.append(sample_mel)
            generated_mel_len.append(sample_mel.shape[2])

        max_generated_mel_len = max(generated_mel_len)
        mel_batch = torch.zeros(
            len(generated_mels),
            flow_module.output_size,
            max_generated_mel_len,
            device=self.model.device,
            dtype=flow_feat.dtype,
        )
        for batch_index, sample_mel in enumerate(generated_mels):
            mel_batch[batch_index, :, : sample_mel.shape[2]] = sample_mel[0]
        speech_batch, _ = self.model.hift.inference(speech_feat=mel_batch)
        sample_per_frame = self._infer_sample_per_mel_frame()
        speech_batch_cpu = speech_batch.cpu()

        output_speeches: List[torch.Tensor] = []
        for batch_index, mel_length in enumerate(generated_mel_len):
            expected_speech_len = int(mel_length * sample_per_frame)
            expected_speech_len = min(expected_speech_len, speech_batch_cpu.shape[1])
            output_speeches.append(
                speech_batch_cpu[batch_index : batch_index + 1, :expected_speech_len]
            )
        return output_speeches

    def _batched_cfm_decode(
        self,
        decoder,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        streaming: bool,
    ) -> torch.Tensor:
        batch_size = mu.shape[0]
        n_timesteps = self.flow_n_timesteps
        z = torch.randn_like(mu).to(mu.device).to(mu.dtype)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if getattr(decoder, "t_scheduler", "") == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # Classifier-free guidance inputs are invariant across diffusion steps.
        mask_in = torch.cat([mask, mask], dim=0)
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        x = z
        t = t_span[0].unsqueeze(0)
        dt = t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            x_in = torch.cat([x, x], dim=0)
            t_in = t.repeat(2 * batch_size).to(spks.dtype)

            dphi_dt = decoder.forward_estimator(
                x_in,
                mask_in,
                mu_in,
                t_in,
                spks_in,
                cond_in,
                streaming=streaming,
            )
            dphi_dt_main, dphi_dt_cfg = torch.split(dphi_dt, [batch_size, batch_size], dim=0)
            dphi_dt = (1.0 + decoder.inference_cfg_rate) * dphi_dt_main - decoder.inference_cfg_rate * dphi_dt_cfg
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x.float()

    def _infer_sample_per_mel_frame(self) -> int:
        hift = self.model.hift
        if hasattr(hift, "upsample_rates") and hasattr(hift, "istft_params"):
            return int(np.prod(hift.upsample_rates) * hift.istft_params["hop_len"])
        if self.cosyvoice.sample_rate == 24000:
            return 480
        return 256

    @staticmethod
    def _concat_segments(segments: List[torch.Tensor]) -> torch.Tensor:
        if not segments:
            raise RuntimeError("No generated speech segments returned by model")
        if len(segments) == 1:
            return segments[0]
        return torch.cat(segments, dim=1)
