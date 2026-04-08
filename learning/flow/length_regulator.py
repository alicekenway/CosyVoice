from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import make_pad_mask


class InterpolateRegulator(nn.Module):
    def __init__(
        self,
        channels: int,
        sampling_ratios: Tuple[int, ...],
        out_channels: int | None = None,
        groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        layers = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for _ in sampling_ratios:
                layers.extend(
                    [
                        nn.Conv1d(channels, channels, 3, 1, 1),
                        nn.GroupNorm(groups, channels),
                        nn.Mish(),
                    ]
                )
        layers.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, ylens: torch.Tensor | None = None):
        mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=int(ylens.max().item()), mode="linear")
        out = self.model(x).transpose(1, 2).contiguous()
        return out * mask, ylens

    def inference(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mel_len1: int,
        mel_len2: int,
        input_frame_rate: int = 50,
    ):
        if x2.shape[1] > 40:
            overlap = int(20 / input_frame_rate * 22050 / 256)
            x2_head = F.interpolate(x2[:, :20].transpose(1, 2).contiguous(), size=overlap, mode="linear")
            x2_mid = F.interpolate(
                x2[:, 20:-20].transpose(1, 2).contiguous(),
                size=mel_len2 - overlap * 2,
                mode="linear",
            )
            x2_tail = F.interpolate(x2[:, -20:].transpose(1, 2).contiguous(), size=overlap, mode="linear")
            x2 = torch.cat([x2_head, x2_mid, x2_tail], dim=2)
        else:
            x2 = F.interpolate(x2.transpose(1, 2).contiguous(), size=mel_len2, mode="linear")

        if x1.shape[1] != 0:
            x1 = F.interpolate(x1.transpose(1, 2).contiguous(), size=mel_len1, mode="linear")
            x = torch.cat([x1, x2], dim=2)
        else:
            x = x2

        out = self.model(x).transpose(1, 2).contiguous()
        return out, mel_len1 + mel_len2
