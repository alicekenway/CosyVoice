from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass(frozen=True)
class TsvInputRow:
    output_index: int
    row_id: str
    text: str
    ref_audio_path: str


@dataclass
class SegmentInput:
    normalized_text: str
    model_input: Dict[str, torch.Tensor]


@dataclass
class PreparedRow:
    row: TsvInputRow
    text_for_metadata: str
    segment_inputs: List[SegmentInput] = field(default_factory=list)


@dataclass
class SynthesisResult:
    row: TsvInputRow
    text_for_metadata: str
    speech: torch.Tensor | None = None
    error_message: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error_message is None and self.speech is not None
