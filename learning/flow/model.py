import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from .decoder import ConditionalDecoder
from .encoder import ConformerEncoder
from .flow_matching import CFMParams, ConditionalCFM
from .length_regulator import InterpolateRegulator
from .utils import make_pad_mask


class MaskedDiffWithXvec(nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        encoder: nn.Module | None = None,
        length_regulator: nn.Module | None = None,
        decoder: nn.Module | None = None,
    ):
        super().__init__()
        if encoder is None or length_regulator is None or decoder is None:
            raise ValueError("encoder, length_regulator, and decoder must be provided")

        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        self.only_mask_loss = only_mask_loss

        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(self.encoder.output_size(), output_size)
        self.length_regulator = length_regulator
        self.decoder = decoder

    def forward(self, batch: dict, device: torch.device):
        token = batch["speech_token"].to(device)
        token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)
        feat_len = batch["speech_feat_len"].to(device)
        embedding = batch["embedding"].to(device)

        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        token_mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * token_mask

        h, _ = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        h, _ = self.length_regulator(h, feat_len)

        conds = torch.zeros(feat.shape, device=token.device)
        for i, seq_len in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * int(seq_len.item())))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mel_mask = (~make_pad_mask(feat_len)).to(h)
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mel_mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
        )
        return {"loss": loss}

    @torch.inference_mode()
    def inference(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        flow_cache: torch.Tensor | None = None,
        n_timesteps: int = 10,
    ):
        del prompt_feat_len
        assert token.shape[0] == 1

        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        prompt_token_count = prompt_token.shape[1]
        token_count = token.shape[1]
        token = torch.cat([prompt_token, token], dim=1)
        token_len = prompt_token_len + token_len
        token_mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * token_mask

        h, _ = self.encoder(token, token_len)
        h = self.encoder_proj(h)

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = int(token_count / self.input_frame_rate * 22050 / 256)
        h, _ = self.length_regulator.inference(
            h[:, :prompt_token_count],
            h[:, prompt_token_count:],
            mel_len1,
            mel_len2,
            self.input_frame_rate,
        )

        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device, dtype=h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mel_mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2], device=token.device))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            prompt_len=mel_len1,
            cache=flow_cache,
        )
        feat = feat[:, :, mel_len1:]
        return feat.float(), flow_cache


def build_cosyvoice1_flow_model() -> MaskedDiffWithXvec:
    encoder = ConformerEncoder(
        input_size=512,
        output_size=512,
        attention_heads=8,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        normalize_before=True,
        input_layer="linear",
        pos_enc_layer_type="rel_pos_espnet",
        selfattention_layer_type="rel_selfattn",
        use_cnn_module=False,
        macaron_style=False,
    )
    length_regulator = InterpolateRegulator(channels=80, sampling_ratios=(1, 1, 1, 1))
    estimator = ConditionalDecoder(
        in_channels=320,
        out_channels=80,
        channels=(256, 256),
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
    )
    decoder = ConditionalCFM(
        in_channels=240,
        n_spks=1,
        spk_emb_dim=80,
        cfm_params=CFMParams(),
        estimator=estimator,
    )
    return MaskedDiffWithXvec(
        input_size=512,
        output_size=80,
        spk_embed_dim=192,
        output_type="mel",
        vocab_size=4096,
        input_frame_rate=50,
        only_mask_loss=True,
        encoder=encoder,
        length_regulator=length_regulator,
        decoder=decoder,
    )
