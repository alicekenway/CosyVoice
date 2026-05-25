from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence

from batch_types import PreparedRow, SegmentInput, TsvInputRow


@dataclass
class PromptFeatures:
    flow_prompt_speech_token: torch.Tensor
    flow_prompt_speech_token_len: torch.Tensor
    prompt_speech_feat: torch.Tensor
    prompt_speech_feat_len: torch.Tensor
    llm_embedding: torch.Tensor
    flow_embedding: torch.Tensor


def collate_segment_inputs(segment_inputs: List[SegmentInput]) -> Dict[str, torch.Tensor]:
    if not segment_inputs:
        raise ValueError("segment_inputs is empty")

    text_list = [item.model_input["text"].squeeze(0).to(torch.int32) for item in segment_inputs]
    text_len_list = [item.model_input["text_len"].squeeze(0).to(torch.int32) for item in segment_inputs]

    prompt_speech_token_list = [
        item.model_input["flow_prompt_speech_token"].squeeze(0).to(torch.int32)
        for item in segment_inputs
    ]
    prompt_speech_token_len_list = [
        item.model_input["flow_prompt_speech_token_len"].squeeze(0).to(torch.int32)
        for item in segment_inputs
    ]

    prompt_speech_feat_list = [
        item.model_input["prompt_speech_feat"].squeeze(0) for item in segment_inputs
    ]
    prompt_speech_feat_len_list = [
        item.model_input["prompt_speech_feat_len"].squeeze(0).to(torch.int32)
        for item in segment_inputs
    ]

    llm_embedding = torch.cat([item.model_input["llm_embedding"] for item in segment_inputs], dim=0)
    flow_embedding = torch.cat([item.model_input["flow_embedding"] for item in segment_inputs], dim=0)

    text = pad_sequence(text_list, batch_first=True, padding_value=0)
    text_len = torch.tensor([length.item() for length in text_len_list], dtype=torch.int32)

    flow_prompt_speech_token = pad_sequence(
        prompt_speech_token_list,
        batch_first=True,
        padding_value=0,
    )
    flow_prompt_speech_token_len = torch.tensor(
        [length.item() for length in prompt_speech_token_len_list],
        dtype=torch.int32,
    )

    prompt_speech_feat = pad_sequence(
        prompt_speech_feat_list,
        batch_first=True,
        padding_value=0.0,
    )
    prompt_speech_feat_len = torch.tensor(
        [length.item() for length in prompt_speech_feat_len_list],
        dtype=torch.int32,
    )

    return {
        "text": text,
        "text_len": text_len,
        "flow_prompt_speech_token": flow_prompt_speech_token,
        "flow_prompt_speech_token_len": flow_prompt_speech_token_len,
        "prompt_speech_feat": prompt_speech_feat,
        "prompt_speech_feat_len": prompt_speech_feat_len,
        "llm_embedding": llm_embedding,
        "flow_embedding": flow_embedding,
    }


class CrossLingualBatchPreparer:
    def __init__(
        self,
        frontend,
        sample_rate: int,
        text_frontend: bool,
        lang_token: str,
    ):
        self.frontend = frontend
        self.sample_rate = sample_rate
        self.text_frontend = text_frontend
        self.lang_token = lang_token
        self._prompt_feature_cache: Dict[str, PromptFeatures] = {}

    def prepare_rows(self, rows: List[TsvInputRow]) -> List[PreparedRow]:
        prepared_rows: List[PreparedRow] = []
        for row in rows:
            prepared_rows.append(self._prepare_row(row))
        return prepared_rows

    def _prepare_row(self, row: TsvInputRow) -> PreparedRow:
        text_for_metadata = row.text
        if self.lang_token and not text_for_metadata.startswith(self.lang_token):
            text_for_metadata = f"{self.lang_token}{text_for_metadata}"

        normalized_texts = self.frontend.text_normalize(
            text_for_metadata,
            split=True,
            text_frontend=self.text_frontend,
        )
        if not normalized_texts:
            raise RuntimeError(f"Text normalization returned empty segments: row_id={row.row_id}")

        prompt_features = self._get_prompt_features(row.ref_audio_path)
        segment_inputs: List[SegmentInput] = []
        for normalized_text in normalized_texts:
            text, text_len = self.frontend._extract_text_token(normalized_text)
            model_input = {
                "text": text,
                "text_len": text_len,
                "flow_prompt_speech_token": prompt_features.flow_prompt_speech_token,
                "flow_prompt_speech_token_len": prompt_features.flow_prompt_speech_token_len,
                "prompt_speech_feat": prompt_features.prompt_speech_feat,
                "prompt_speech_feat_len": prompt_features.prompt_speech_feat_len,
                "llm_embedding": prompt_features.llm_embedding,
                "flow_embedding": prompt_features.flow_embedding,
            }
            segment_inputs.append(
                SegmentInput(
                    normalized_text=normalized_text,
                    model_input=model_input,
                )
            )

        return PreparedRow(
            row=row,
            text_for_metadata=text_for_metadata,
            segment_inputs=segment_inputs,
        )

    def _get_prompt_features(self, ref_audio_path: str) -> PromptFeatures:
        resolved_audio_path = str(Path(ref_audio_path).expanduser().resolve())
        if resolved_audio_path in self._prompt_feature_cache:
            return self._prompt_feature_cache[resolved_audio_path]

        prompt_speech_feat, prompt_speech_feat_len = self.frontend._extract_speech_feat(resolved_audio_path)
        flow_prompt_speech_token, flow_prompt_speech_token_len = self.frontend._extract_speech_token(
            resolved_audio_path
        )
        if self.sample_rate == 24000:
            # Keep the same token/feature alignment as the original frontend_zero_shot logic.
            token_len = min(
                int(prompt_speech_feat.shape[1] / 2),
                flow_prompt_speech_token.shape[1],
            )
            prompt_speech_feat = prompt_speech_feat[:, : 2 * token_len]
            prompt_speech_feat_len = torch.tensor([2 * token_len], dtype=torch.int32).to(
                prompt_speech_feat.device
            )
            flow_prompt_speech_token = flow_prompt_speech_token[:, :token_len]
            flow_prompt_speech_token_len = torch.tensor([token_len], dtype=torch.int32).to(
                flow_prompt_speech_token.device
            )

        embedding = self.frontend._extract_spk_embedding(resolved_audio_path)
        prompt_features = PromptFeatures(
            flow_prompt_speech_token=flow_prompt_speech_token,
            flow_prompt_speech_token_len=flow_prompt_speech_token_len,
            prompt_speech_feat=prompt_speech_feat,
            prompt_speech_feat_len=prompt_speech_feat_len,
            llm_embedding=embedding,
            flow_embedding=embedding,
        )
        self._prompt_feature_cache[resolved_audio_path] = prompt_features
        return prompt_features
