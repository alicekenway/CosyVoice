from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CFMParams:
    sigma_min: float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "cosine"
    training_cfg_rate: float = 0.2
    inference_cfg_rate: float = 0.7
    reg_loss_type: str = "l1"


class ConditionalCFM(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        cfm_params: CFMParams,
        n_spks: int = 1,
        spk_emb_dim: int = 64,
        estimator: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.n_feats = in_channels
        self.sigma_min = cfm_params.sigma_min
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.estimator = estimator

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        prompt_len: int = 0,
        cache: torch.Tensor | None = None,
    ):
        z = torch.randn_like(mu) * temperature
        if cache is not None and cache.numel() > 0:
            cache_size = cache.shape[2]
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        else:
            cache_size = 0

        if cache_size != 0 or prompt_len > 0:
            z_cache = torch.cat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
            mu_cache = torch.cat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
            new_cache = torch.stack([z_cache, mu_cache], dim=-1)
        else:
            new_cache = torch.zeros(
                mu.shape[0], mu.shape[1], 0, 2, device=mu.device, dtype=mu.dtype
            )

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span, mu, mask, spks, cond), new_cache

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor | None,
        cond: torch.Tensor | None,
    ) -> torch.Tensor:
        t = t_span[0].unsqueeze(0)
        dt = t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            if spks is None:
                raise ValueError("spks must be provided for CosyVoice1 flow inference")

            x_in = x.repeat(2, 1, 1)
            mask_in = mask.repeat(2, 1, 1)
            mu_in = torch.zeros_like(x_in)
            mu_in[0] = mu[0]
            t_in = t.repeat(2).to(spks.dtype)
            spks_in = torch.zeros(2, spks.shape[1], device=spks.device, dtype=spks.dtype)
            spks_in[0] = spks[0]
            cond_in = torch.zeros_like(x_in)
            if cond is not None:
                cond_in[0] = cond[0]

            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x.float()

    def compute_loss(
        self,
        x1: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ):
        b, _, _ = mu.shape
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1) if spks is not None else None
            cond = cond * cfg_mask.view(-1, 1, 1) if cond is not None else None

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y
