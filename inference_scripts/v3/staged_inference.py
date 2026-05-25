from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import warnings
from torch.nn.utils.rnn import pad_sequence

from batch_types import PreparedRow, SegmentInput, SynthesisResult
from frontend_batch import collate_segment_inputs
from io_utils import chunked


@dataclass(frozen=True)
class SegmentTask:
    row_index: int
    segment_index: int
    segment_input: SegmentInput


def _reindex_kv_cache(cache, keep_indices: torch.Tensor):
    """Select a subset of batch entries from a HuggingFace KV cache."""
    if cache is None:
        return None
    # DynamicCache (transformers >= 4.36) has reorder_cache
    if hasattr(cache, "reorder_cache"):
        cache.reorder_cache(keep_indices)
        return cache
    # Legacy tuple-of-tuples: ((key_0, value_0), (key_1, value_1), ...)
    if isinstance(cache, tuple):
        return tuple(
            (k.index_select(0, keep_indices), v.index_select(0, keep_indices))
            for k, v in cache
        )
    raise TypeError(f"Unsupported KV cache type: {type(cache)}")


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
    ):
        self.cosyvoice = cosyvoice
        self.model = cosyvoice.model
        self.on_error = on_error
        self.llm_batch_size = llm_batch_size
        self.flow_batch_size = flow_batch_size
        self.min_token_text_ratio = min_token_text_ratio
        self.max_token_text_ratio = max_token_text_ratio
        self.flow_n_timesteps = flow_n_timesteps

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
                    [task.segment_input.model_input for task in active_tasks]
                )
            except Exception as exc:
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
            except Exception as exc:
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

    # ------------------------------------------------------------------
    # LLM stage
    # ------------------------------------------------------------------

    def _run_llm_stage_batch(
        self, model_inputs: Sequence[Dict[str, torch.Tensor]]
    ) -> List[List[int]]:
        batch_inputs = collate_segment_inputs(
            [SegmentInput(normalized_text="", model_input=item) for item in model_inputs]
        )

        llm_module = self.model.llm
        if not hasattr(llm_module, "llm") or not hasattr(llm_module, "llm_decoder"):
            raise RuntimeError("Current LLM module does not support strict batch decode")
        if not hasattr(llm_module, "speech_embedding"):
            raise RuntimeError("Current LLM module does not expose speech_embedding")

        device = self.model.device
        text = batch_inputs["text"].to(device)
        text_len = batch_inputs["text_len"].to(device)
        batch_size = text.shape[0]

        # Prepare text encodings based on backend type
        is_transformer_backend = hasattr(llm_module, "text_embedding") and hasattr(
            llm_module, "encode"
        )
        is_qwen_backend = (
            hasattr(llm_module.llm, "model")
            and hasattr(llm_module.llm.model, "model")
            and hasattr(llm_module.llm.model.model, "embed_tokens")
        )

        if is_transformer_backend:
            token_table = llm_module.llm_embedding.weight
            text_emb = llm_module.text_embedding(text.to(torch.long))
            text_encoded, text_encoded_len = llm_module.encode(text_emb, text_len)
            llm_embedding = batch_inputs["llm_embedding"].to(device)
            llm_embedding = F.normalize(llm_embedding, dim=1)
            llm_embedding = llm_module.spk_embed_affine_layer(llm_embedding).unsqueeze(1)
        elif is_qwen_backend:
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

        # Build per-sample prefix sequences
        prefix_sequences: List[torch.Tensor] = []
        prefix_lengths: List[int] = []
        for batch_index in range(batch_size):
            effective_len = int(text_encoded_len[batch_index].item())
            parts = [sos_emb.squeeze(0)]
            if llm_embedding is not None:
                parts.append(llm_embedding[batch_index : batch_index + 1].squeeze(0))
            parts.append(
                text_encoded[batch_index : batch_index + 1, :effective_len, :].squeeze(0)
            )
            parts.append(task_id_emb.squeeze(0))
            prefix = torch.cat(parts, dim=0)
            prefix_sequences.append(prefix)
            prefix_lengths.append(prefix.shape[0])

        min_len = (text_len.float() * self.min_token_text_ratio).long()
        max_len = (text_len.float() * self.max_token_text_ratio).long()
        max_decode_steps = int(max_len.max().item())
        stop_token_ids = getattr(llm_module, "stop_token_ids", [llm_module.eos_token])
        stop_token_set = set(int(token_id) for token_id in stop_token_ids)

        # Use KV-cached decoding when available (Qwen2 path); fall back to
        # full-sequence forward otherwise (TransformerLM path).
        has_kv_cache = hasattr(llm_module.llm, "forward_one_step")
        if has_kv_cache:
            generated_tokens = self._llm_decode_cached(
                llm_module, prefix_sequences, prefix_lengths,
                min_len, max_len, max_decode_steps, batch_size,
                stop_token_set, device,
            )
        else:
            warnings.warn(
                "LLM backend does not support KV-cache decoding (missing "
                "`forward_one_step`). Falling back to full-sequence decoding, "
                "which can add significant latency. "
                f"llm_module={llm_module.__class__.__name__}, "
                f"llm_encoder={llm_module.llm.__class__.__name__}",
                RuntimeWarning,
            )
            generated_tokens = self._llm_decode_full_seq(
                llm_module, prefix_sequences, prefix_lengths,
                min_len, max_len, max_decode_steps, batch_size,
                stop_token_set, device,
            )

        for token_sequence in generated_tokens:
            if not token_sequence:
                raise RuntimeError(
                    "LLM generated zero speech tokens for at least one sample"
                )
        return generated_tokens

    def _llm_decode_cached(
        self,
        llm_module,
        prefix_sequences: List[torch.Tensor],
        prefix_lengths: List[int],
        min_len: torch.Tensor,
        max_len: torch.Tensor,
        max_decode_steps: int,
        batch_size: int,
        stop_token_set: set,
        device: torch.device,
    ) -> List[List[int]]:
        """Autoregressive decode with KV cache (Qwen2 / CosyVoice3 path).

        Only the new token embedding is fed at each step; previous context is
        retrieved from the KV cache, avoiding O(T^2) recomputation per step.
        Prefixes are left-padded so that ``y_pred[:, -1]`` always corresponds
        to the last valid position for every sample in the batch.
        """
        max_prefix_len = max(prefix_lengths)
        hidden_dim = prefix_sequences[0].shape[1]
        prefix_lens = torch.tensor(prefix_lengths, device=device, dtype=torch.long)

        # Left-pad prefixes so every sample's last token is at position max_prefix_len-1.
        prefix_buffer = torch.zeros(
            batch_size, max_prefix_len, hidden_dim,
            device=device, dtype=prefix_sequences[0].dtype,
        )
        for batch_index, prefix in enumerate(prefix_sequences):
            pad_offset = max_prefix_len - prefix.shape[0]
            prefix_buffer[batch_index, pad_offset:] = prefix

        # Pre-compute attention mask for the maximum possible total length.
        # Left-padded layout: positions >= pad_offset are valid.
        max_total_len = max_prefix_len + max_decode_steps
        positions = torch.arange(max_total_len, device=device)
        pad_offsets = max_prefix_len - prefix_lens  # (batch,)
        attn_mask_full = positions.unsqueeze(0) >= pad_offsets.unsqueeze(1)

        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        active_global_indices = list(range(batch_size))
        active_attn_mask = attn_mask_full

        cache = None
        step_input = prefix_buffer

        for step in range(max_decode_steps):
            if not active_global_indices:
                break

            total_seq_len = max_prefix_len + step
            mask_slice = active_attn_mask[:, :total_seq_len]
            # forward_one_step expects (batch, ?, seq_len) and takes masks[:, -1, :]
            masks_3d = mask_slice.unsqueeze(1)

            y_pred, cache = llm_module.llm.forward_one_step(
                step_input, masks_3d, cache
            )

            last_hidden = y_pred[:, -1]
            logp = llm_module.llm_decoder(last_hidden).log_softmax(dim=-1)

            keep_local_indices: List[int] = []
            next_token_embeds: List[torch.Tensor] = []

            for local_idx, global_idx in enumerate(active_global_indices):
                ignore_eos = step < int(min_len[global_idx].item())
                top_token = llm_module.sampling_ids(
                    logp[local_idx],
                    generated_tokens[global_idx],
                    25,
                    ignore_eos=ignore_eos,
                )
                top_token_int = (
                    int(top_token.item())
                    if isinstance(top_token, torch.Tensor)
                    else int(top_token)
                )
                if top_token_int in stop_token_set:
                    continue
                generated_tokens[global_idx].append(top_token_int)
                if len(generated_tokens[global_idx]) < int(max_len[global_idx].item()):
                    keep_local_indices.append(local_idx)
                    next_token_embeds.append(
                        llm_module.speech_embedding.weight[
                            top_token_int : top_token_int + 1
                        ]
                    )

            if not keep_local_indices:
                break

            step_input = torch.stack(next_token_embeds, dim=0)

            if len(keep_local_indices) < len(active_global_indices):
                keep_tensor = torch.tensor(
                    keep_local_indices, device=device, dtype=torch.long
                )
                cache = _reindex_kv_cache(cache, keep_tensor)
                active_attn_mask = active_attn_mask.index_select(0, keep_tensor)

            active_global_indices = [
                active_global_indices[i] for i in keep_local_indices
            ]

        return generated_tokens

    def _llm_decode_full_seq(
        self,
        llm_module,
        prefix_sequences: List[torch.Tensor],
        prefix_lengths: List[int],
        min_len: torch.Tensor,
        max_len: torch.Tensor,
        max_decode_steps: int,
        batch_size: int,
        stop_token_set: set,
        device: torch.device,
    ) -> List[List[int]]:
        """Fallback decode without KV cache (TransformerLM path).

        The full prefix + generated tokens are re-fed at every step.  This is
        correct but O(T^3) in total attention cost.  Prefer ``_llm_decode_cached``
        when the underlying encoder supports ``forward_one_step``.
        """
        max_prefix_len = max(prefix_lengths)
        hidden_dim = prefix_sequences[0].shape[1]
        max_sequence_len = max_prefix_len + max_decode_steps

        sequence_buffer = torch.zeros(
            batch_size, max_sequence_len, hidden_dim,
            device=device, dtype=prefix_sequences[0].dtype,
        )
        for batch_index, prefix in enumerate(prefix_sequences):
            sequence_buffer[batch_index, : prefix.shape[0]] = prefix
        sequence_len = torch.tensor(
            prefix_lengths, dtype=torch.long, device=device
        )

        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        active_sample_indices = torch.arange(
            batch_size, device=device, dtype=torch.long
        )

        for step in range(max_decode_steps):
            if active_sample_indices.numel() == 0:
                break

            active_input = sequence_buffer.index_select(0, active_sample_indices)
            active_len = sequence_len.index_select(0, active_sample_indices)
            active_max_len = int(active_len.max().item())
            active_input = active_input[:, :active_max_len, :]

            y_pred, _ = llm_module.llm(active_input, active_len)

            gather_index = (
                (active_len - 1)
                .view(-1, 1, 1)
                .expand(-1, 1, y_pred.shape[-1])
            )
            last_hidden = y_pred.gather(1, gather_index).squeeze(1)
            logp = llm_module.llm_decoder(last_hidden).log_softmax(dim=-1)

            next_active: List[int] = []
            for local_idx in range(active_sample_indices.shape[0]):
                global_idx = int(active_sample_indices[local_idx].item())
                ignore_eos = step < int(min_len[global_idx].item())
                top_token = llm_module.sampling_ids(
                    logp[local_idx],
                    generated_tokens[global_idx],
                    25,
                    ignore_eos=ignore_eos,
                )
                top_token_int = (
                    int(top_token.item())
                    if isinstance(top_token, torch.Tensor)
                    else int(top_token)
                )
                if top_token_int in stop_token_set:
                    continue
                generated_tokens[global_idx].append(top_token_int)
                next_embed = llm_module.speech_embedding.weight[
                    top_token_int : top_token_int + 1
                ]
                write_pos = int(sequence_len[global_idx].item())
                sequence_buffer[global_idx, write_pos : write_pos + 1] = next_embed
                sequence_len[global_idx] = write_pos + 1
                if len(generated_tokens[global_idx]) < int(max_len[global_idx].item()):
                    next_active.append(global_idx)

            active_sample_indices = torch.tensor(
                next_active, dtype=torch.long, device=device
            )

        return generated_tokens

    # ------------------------------------------------------------------
    # Flow + HiFi-GAN stage
    # ------------------------------------------------------------------

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
            raise RuntimeError("token_ids_list contains an empty sequence")

        from cosyvoice.utils.mask import make_pad_mask  # pylint: disable=import-outside-toplevel

        device = self.model.device

        gen_token_tensors = [
            torch.tensor(token_ids, dtype=torch.int32) for token_ids in token_ids_list
        ]
        gen_token_len = torch.tensor(
            [t.shape[0] for t in gen_token_tensors],
            dtype=torch.int32, device=device,
        )
        gen_token = pad_sequence(
            gen_token_tensors, batch_first=True, padding_value=0
        ).to(device)

        prompt_token = batch_inputs["flow_prompt_speech_token"].to(device, dtype=torch.int32)
        prompt_token_len = batch_inputs["flow_prompt_speech_token_len"].to(device, dtype=torch.int32)
        prompt_feat = batch_inputs["prompt_speech_feat"].to(device)
        prompt_feat_len = batch_inputs["prompt_speech_feat_len"].to(device, dtype=torch.int32)
        flow_embedding = batch_inputs["flow_embedding"].to(device)

        if not hasattr(flow_module, "input_embedding") or not hasattr(flow_module, "decoder"):
            raise RuntimeError("Current flow module does not support strict batch path")
        if not isinstance(flow_module.decoder.estimator, torch.nn.Module):
            raise RuntimeError(
                "TensorRT flow estimator path is unsupported in strict batch mode"
            )

        embedding = F.normalize(flow_embedding, dim=1)
        embedding = flow_module.spk_embed_affine_layer(embedding)

        # Concatenate prompt + generated tokens per sample, then pad
        token_total_tensors: List[torch.Tensor] = []
        for batch_index in range(prompt_token.shape[0]):
            prompt_len = int(prompt_token_len[batch_index].item())
            gen_len = int(gen_token_len[batch_index].item())
            token_total_tensors.append(
                torch.cat([
                    prompt_token[batch_index, :prompt_len],
                    gen_token[batch_index, :gen_len],
                ], dim=0)
            )
        token_total = pad_sequence(token_total_tensors, batch_first=True, padding_value=0)
        token_total_len = torch.tensor(
            [t.shape[0] for t in token_total_tensors],
            dtype=torch.int32, device=device,
        )

        token_mask = (
            (~make_pad_mask(token_total_len, max_len=token_total.shape[1]))
            .unsqueeze(-1)
            .to(embedding)
        )
        token_emb = flow_module.input_embedding(torch.clamp(token_total, min=0)) * token_mask

        # Encoder + optional length regulator
        if hasattr(flow_module, "length_regulator"):
            h, h_lengths = flow_module.encoder(token_emb, token_total_len)
            h = flow_module.encoder_proj(h)
            mel_new_len = torch.tensor(
                [
                    int(length.item() / flow_module.input_frame_rate * 22050 / 256)
                    for length in gen_token_len
                ],
                dtype=torch.int32, device=device,
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

        max_supported_mel_len = h.shape[1]
        mel_total_len = torch.clamp(mel_total_len, max=max_supported_mel_len)
        max_mel_len = int(mel_total_len.max().item())
        h = h[:, :max_mel_len]

        conds = torch.zeros(
            h.shape[0], max_mel_len, flow_module.output_size,
            device=device, dtype=h.dtype,
        )
        for batch_index in range(h.shape[0]):
            prompt_len = int(prompt_feat_len[batch_index].item())
            conds[batch_index, :prompt_len] = prompt_feat[batch_index, :prompt_len]

        mel_mask = (
            (~make_pad_mask(mel_total_len, max_len=max_mel_len))
            .unsqueeze(1)
            .to(h)
        )
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

        # Extract per-sample generated mel (skip prompt region)
        generated_mels: List[torch.Tensor] = []
        generated_mel_lengths: List[int] = []
        for batch_index in range(flow_feat.shape[0]):
            start = int(prompt_feat_len[batch_index].item())
            end = int(mel_total_len[batch_index].item())
            sample_mel = flow_feat[batch_index : batch_index + 1, :, start:end]
            generated_mels.append(sample_mel)
            generated_mel_lengths.append(sample_mel.shape[2])

        # Pad mels into a batch and run HiFi-GAN vocoder
        max_generated_mel_len = max(generated_mel_lengths)
        mel_batch = torch.zeros(
            len(generated_mels), flow_module.output_size, max_generated_mel_len,
            device=device, dtype=flow_feat.dtype,
        )
        for batch_index, sample_mel in enumerate(generated_mels):
            mel_batch[batch_index, :, : sample_mel.shape[2]] = sample_mel[0]

        speech_batch, _ = self.model.hift.inference(speech_feat=mel_batch)
        sample_per_frame = self._infer_sample_per_mel_frame()
        speech_batch_cpu = speech_batch.cpu()

        output_speeches: List[torch.Tensor] = []
        for batch_index, mel_length in enumerate(generated_mel_lengths):
            expected_speech_len = min(
                int(mel_length * sample_per_frame),
                speech_batch_cpu.shape[1],
            )
            output_speeches.append(
                speech_batch_cpu[batch_index : batch_index + 1, :expected_speech_len]
            )
        return output_speeches

    # ------------------------------------------------------------------
    # CFM (Conditional Flow Matching) ODE solver
    # ------------------------------------------------------------------

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
        z = torch.randn_like(mu)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if getattr(decoder, "t_scheduler", "") == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # Classifier-free guidance inputs (invariant across diffusion steps)
        mask_in = torch.cat([mask, mask], dim=0)
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        # Pre-allocate per-step buffers to avoid repeated allocations
        x_in = torch.empty(
            2 * batch_size, *mu.shape[1:], device=mu.device, dtype=mu.dtype
        )
        t_in = torch.empty(2 * batch_size, device=mu.device, dtype=spks.dtype)

        x = z
        t = t_span[0].unsqueeze(0)
        dt = t_span[1] - t_span[0]

        for step_idx in range(1, len(t_span)):
            x_in[:batch_size] = x
            x_in[batch_size:] = x
            t_in[:] = t

            dphi_dt = decoder.forward_estimator(
                x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=streaming,
            )
            dphi_dt_main, dphi_dt_cfg = dphi_dt.split(
                [batch_size, batch_size], dim=0
            )
            dphi_dt = (
                (1.0 + decoder.inference_cfg_rate) * dphi_dt_main
                - decoder.inference_cfg_rate * dphi_dt_cfg
            )
            x = x + dt * dphi_dt
            t = t + dt
            if step_idx < len(t_span) - 1:
                dt = t_span[step_idx + 1] - t

        return x.float()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
