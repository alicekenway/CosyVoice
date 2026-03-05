from dataclasses import dataclass
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

    def _run_llm_stage_batch(self, model_inputs: Sequence[Dict[str, torch.Tensor]]) -> List[List[int]]:
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

        lm_input_list: List[torch.Tensor] = []
        for batch_index in range(batch_size):
            effective_len = int(text_encoded_len[batch_index].item())
            prefix_list = [sos_emb.squeeze(0)]
            if llm_embedding is not None:
                prefix_list.append(llm_embedding[batch_index : batch_index + 1].squeeze(0))
            lm_input_list.append(
                torch.cat(
                    prefix_list
                    + [
                        text_encoded[batch_index : batch_index + 1, :effective_len, :].squeeze(0),
                        task_id_emb.squeeze(0),
                    ],
                    dim=0,
                )
            )

        lm_input_len = torch.tensor(
            [item.shape[0] for item in lm_input_list],
            dtype=torch.long,
            device=self.model.device,
        )
        min_len = ((text_len.to(torch.float32)) * self.min_token_text_ratio).to(torch.long)
        max_len = ((text_len.to(torch.float32)) * self.max_token_text_ratio).to(torch.long)
        max_decode_steps = int(max_len.max().item())

        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        active_sample_indices = list(range(batch_size))
        stop_token_ids = getattr(llm_module, "stop_token_ids", [llm_module.eos_token])
        stop_token_set = set(int(token_id) for token_id in stop_token_ids)
        current_sequences = lm_input_list

        for decode_step in range(max_decode_steps):
            if not active_sample_indices:
                break

            active_sequences = [current_sequences[sample_index] for sample_index in active_sample_indices]
            padded_input = pad_sequence(active_sequences, batch_first=True, padding_value=0.0)
            padded_len = torch.tensor(
                [sequence.shape[0] for sequence in active_sequences],
                dtype=torch.long,
                device=self.model.device,
            )
            y_pred, _ = llm_module.llm(padded_input, padded_len)
            gather_index = (
                (padded_len - 1)
                .view(len(active_sample_indices), 1, 1)
                .expand(-1, 1, y_pred.shape[-1])
            )
            current_last_hidden = y_pred.gather(1, gather_index).squeeze(1)
            logp = llm_module.llm_decoder(current_last_hidden).log_softmax(dim=-1)
            next_active_sample_indices: List[int] = []
            for active_batch_index, sample_index in enumerate(active_sample_indices):
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
                current_sequences[sample_index] = torch.cat(
                    [current_sequences[sample_index], next_token_embed],
                    dim=0,
                )
                if len(generated_tokens[sample_index]) < int(max_len[sample_index].item()):
                    next_active_sample_indices.append(sample_index)
            active_sample_indices = next_active_sample_indices

        for token_sequence in generated_tokens:
            if not token_sequence:
                raise RuntimeError("LLM generated zero speech tokens for at least one sample")
        return generated_tokens

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

        output_speeches: List[torch.Tensor] = []
        for batch_index, mel_length in enumerate(generated_mel_len):
            expected_speech_len = int(mel_length * sample_per_frame)
            expected_speech_len = min(expected_speech_len, speech_batch.shape[1])
            output_speeches.append(
                speech_batch[batch_index : batch_index + 1, :expected_speech_len].cpu()
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
        x = z
        t = t_span[0].unsqueeze(0)
        dt = t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            x_in = torch.cat([x, x], dim=0)
            mask_in = torch.cat([mask, mask], dim=0)
            mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
            spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
            cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
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
