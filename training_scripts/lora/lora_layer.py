import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank adapter.

    The original weight is frozen. Only lora_A and lora_B are trainable.
    At init lora_B is zero, so the output equals the original linear layer.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # keep the original linear frozen
        self.linear = original_linear
        for p in self.linear.parameters():
            p.requires_grad = False

        # low-rank adapter
        self.lora_A = nn.Parameter(torch.empty(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        if self.merged:
            return base_out
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base_out + lora_out

    def merge(self):
        """Fold LoRA weights into the base linear for faster inference."""
        if not self.merged:
            self.linear.weight.data += (self.lora_B.t() @ self.lora_A.t()) * self.scaling
            self.merged = True

    def unmerge(self):
        """Undo merge so training can continue."""
        if self.merged:
            self.linear.weight.data -= (self.lora_B.t() @ self.lora_A.t()) * self.scaling
            self.merged = False
