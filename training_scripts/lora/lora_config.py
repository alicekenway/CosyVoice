from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    # which part to apply LoRA: "llm", "flow", or "both"
    target: str = 'llm'

    # common LoRA hyperparams
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05

    # LLM-specific: which Qwen2 modules to target
    llm_target_modules: List[str] = field(
        default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )
    train_llm_decoder: bool = True
    train_speech_embedding: bool = True

    # Flow/DiT-specific
    dit_target_attn: bool = True
    dit_target_ff: bool = False

    @staticmethod
    def from_dict(d: dict) -> 'LoRAConfig':
        return LoRAConfig(**{k: v for k, v in d.items() if k in LoRAConfig.__dataclass_fields__})
