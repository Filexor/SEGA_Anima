import comfy
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrapperExecutor, WrappersMP
from comfy_api.latest import io

from comfy.ldm.cosmos.predict2 import MiniTrainDIT
from comfy.ldm.cosmos.position_embedding import VideoRopePosition3DEmb

import functools
import math

from typing import Optional, Union

import torch

from torchvision import transforms
from einops import rearrange, repeat

File_x_SEGA_Anima_key = "File_x_SEGA_Anima_key"
File_x_SEGA_Anima_state = "File_x_SEGA_Anima_state"

def compute_base_mscale(
    target_res: int,
    training_res: int,
    formula: str = "power_res",
    coefficient: Optional[float] = None,
) -> float:
    s = max(float(target_res) / float(training_res), 1.0)
    if formula == "power_res":
        c = 0.1 if coefficient is None else float(coefficient)
        return s ** c
    if formula == "log_res":
        c = 0.1 if coefficient is None else float(coefficient)
        return 1.0 + c * math.log(s)
    raise ValueError(f"Unknown base_mscale formula: {formula!r}. Use 'power_res' or 'log_res'.")

@torch.no_grad()
def compute_sega_allocation(
    energy_profile: torch.Tensor,
    freqs: torch.Tensor,
    base_mscale: float,
    spread: float,
    alpha: float = 0.15,
    beta: float = 1.5,
    min_mscale: float = 1.0,
) -> torch.Tensor:
    """
    Spectral-Energy Guided Attention (SEGA): per-dim RoPE mscale (asymmetric band).

        z_k = standardise( log E[bin(k)] )
        s_k = tanh( beta * z_k ),  re-centred
        m_k = base_mscale * ( 1 - alpha * spread * s_k )

    High spectral-energy dims get lower m_k (sharpness-biased).
    """
    D_half = freqs.shape[0]
    eps = 1e-8

    if spread <= 0.0 or alpha <= 0.0:
        return torch.full((D_half,), float(base_mscale), device=freqs.device, dtype=torch.float32)

    # Map each RoPE dim to its FFT bin via log-period
    periods = 2.0 * math.pi / freqs.clamp(min=eps)
    log_periods = torch.log(periods)
    min_lp, max_lp = log_periods.min(), log_periods.max()
    if (max_lp - min_lp).item() > 1e-6:
        lp_norm = (log_periods - min_lp) / (max_lp - min_lp)  # 0=high-freq, 1=low-freq
    else:
        lp_norm = torch.zeros_like(log_periods)

    n_bins = energy_profile.shape[0]
    bin_pos = (1.0 - lp_norm) * (n_bins - 1)
    j_low = bin_pos.floor().long().clamp(0, n_bins - 1)
    j_high = (j_low + 1).clamp(0, n_bins - 1)
    frac = (bin_pos - j_low.to(bin_pos.dtype)).clamp(0.0, 1.0)

    E = energy_profile.to(freqs.device).clamp(min=eps)
    log_E = torch.log(E)
    raw = log_E[j_low] * (1.0 - frac) + log_E[j_high] * frac

    # Standardise + tanh + re-centre
    z = raw - raw.mean()
    z = z / z.std().clamp(min=eps)
    s = torch.tanh(float(beta) * z)
    s = s - s.mean()

    # direction = -1  =>  subtract
    m = float(base_mscale) * (1.0 - float(alpha) * float(spread) * s)
    return m.clamp(min=float(min_mscale)).to(torch.float32)

class File_x_SEGA_Anima_RoPE(torch.nn.Module):
    def __init__(
        self, 
        axes_dims_rope: tuple[int, int, int] = (44, 42, 42),
        theta: float = 10000.0,
        base_mscale_formula: str = "power_res",
        base_mscale_coefficient: Optional[float] = None,
        training_resolution_h: float = 1024,
        training_resolution_w: float = 1024,
        mscale_alpha: float = 0.15,
        mscale_beta: float = 1.5,
        mscale_min: float = 1.0,
        debug: int = 0b0,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.axes_dim = axes_dims_rope
        self.theta = theta
        self.base_mscale_formula = base_mscale_formula
        self.base_mscale_coefficient = base_mscale_coefficient
        self.training_resolution_h = training_resolution_h
        self.training_resolution_w = training_resolution_w
        self.mscale_alpha = mscale_alpha
        self.mscale_beta = mscale_beta
        self.mscale_min = mscale_min
        self.frame = 0
        self.height = 64
        self.width = 64
        self.build_freqs_with_size(self.frame, self.height, self.width, 1.0, 1.0, 1.0, debug)

    @staticmethod
    def rope_params(index, dim, theta=10000, ntk_factor=1.0, debug=0b0):
        assert dim % 2 == 0
        scaled_theta = theta * ntk_factor
        freqs = torch.outer(index, 1.0 / torch.pow(scaled_theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        # freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def build_freqs_with_size(self, frame, height, width, ntk_t: float, ntk_h: float, ntk_w: float, debug:int):
        """(Re)build pos/neg frequency tensors for the given per-axis NTK factors."""
        pos_index_frame = torch.arange(frame, dtype=torch.float)
        pos_index_height = torch.arange(height, dtype=torch.float)
        pos_index_width = torch.arange(width, dtype=torch.float)
        neg_index_frame = torch.arange(frame, dtype=torch.float).flip(0) * -1 - 1
        neg_index_height = torch.arange(height, dtype=torch.float).flip(0) * -1 - 1
        neg_index_width = torch.arange(width, dtype=torch.float).flip(0) * -1 - 1

        self.pos_freqs_t = self.rope_params(pos_index_frame, self.axes_dim[0], self.theta, ntk_factor=ntk_t, debug=debug)
        self.pos_freqs_h = self.rope_params(pos_index_height, self.axes_dim[1], self.theta, ntk_factor=ntk_h, debug=debug)
        self.pos_freqs_w = self.rope_params(pos_index_width, self.axes_dim[2], self.theta, ntk_factor=ntk_w, debug=debug)

        self.neg_freqs_t = self.rope_params(neg_index_frame, self.axes_dim[0], self.theta, ntk_factor=ntk_t, debug=debug)
        self.neg_freqs_h = self.rope_params(neg_index_height, self.axes_dim[1], self.theta, ntk_factor=ntk_h, debug=debug)
        self.neg_freqs_w = self.rope_params(neg_index_width, self.axes_dim[2], self.theta, ntk_factor=ntk_w, debug=debug)

        if ntk_t > 1.0:
            scaled_theta_t = self.theta * ntk_t
            temporal_dim = self.axes_dim[0]
            self._inv_freqs_t = 1.0 / torch.pow(
                scaled_theta_t, torch.arange(0, temporal_dim, 2).float().div(temporal_dim)
            )
        if ntk_h > 1.0:
            scaled_theta_h = self.theta * ntk_h
            spatial_dim = self.axes_dim[1]
            self._inv_freqs_h = 1.0 / torch.pow(
                scaled_theta_h, torch.arange(0, spatial_dim, 2).float().div(spatial_dim)
            )
        if ntk_w > 1.0:
            scaled_theta_w = self.theta * ntk_w
            spatial_dim = self.axes_dim[2]
            self._inv_freqs_w = 1.0 / torch.pow(
                scaled_theta_w, torch.arange(0, spatial_dim, 2).float().div(spatial_dim)
            )

        self.ntk_t = ntk_t
        self.ntk_h = ntk_h
        self.ntk_w = ntk_w

        if hasattr(self, 'compute_condition_freqs'):
            self.compute_condition_freqs.cache_clear()

    def ensure_ntk_factor(self, frame: int, height: int, width: int, ntk_t: float, ntk_h: float, ntk_w: float, debug: int):
        """Rebuild frequencies if per-axis NTK factors changed."""
        if (frame != self.frame or
            height != self.height or
            width != self.width or
            abs(ntk_t - self.ntk_t) > 1e-8 or 
            abs(ntk_h - self.ntk_h) > 1e-8 or
            abs(ntk_w - self.ntk_w) > 1e-8):
            self.build_freqs_with_size(frame, height, width, ntk_t, ntk_h, ntk_w, debug)

    @torch.no_grad()
    def compute_mscale(self, ntk_factor_t, ntk_factor_h, ntk_factor_w, mscale_spread,
                       energy_profile_t=None, energy_profile_h=None, energy_profile_w=None,
                       target_res_h=None, target_res_w=None, device=None, debug=0b0):
        """
        Compute per-frequency-dimension mscale with SEGA (Spectral-Energy Guided Attention).
        Returns None if no scaling needed, otherwise [D_total/2] real tensor.
        """
        if (ntk_factor_h <= 1.0 and ntk_factor_w <= 1.0) or mscale_spread is None:
            return None

        # temporal axis SEGA
        if ntk_factor_t > 1.0 and hasattr(self, '_inv_freqs_t'):
            base_ms_t = compute_base_mscale(
                1.0,
                1.0,
                self.base_mscale_formula,
                self.base_mscale_coefficient,
            )
            dev = device or self._inv_freqs_t.device
            mscale_t = torch.full((self.axes_dim[0] // 2,), base_ms_t,
                                       device=dev, dtype=torch.float32)
        else:
            mscale_t = None

        # Height axis SEGA
        if ntk_factor_h > 1.0 and hasattr(self, '_inv_freqs_h'):
            base_ms_h = compute_base_mscale(
                target_res_h or self.training_resolution_h,
                self.training_resolution_h,
                self.base_mscale_formula,
                self.base_mscale_coefficient,
            )
            use_sega_h = (
                energy_profile_h is not None
                and float(mscale_spread) > 0.0
                and base_ms_h > 1.0 + 1e-8
            )
            if use_sega_h:
                inv_f = self._inv_freqs_h.to(device) if device is not None else self._inv_freqs_h
                mscale_h = compute_sega_allocation(
                    energy_profile_h, inv_f,
                    base_ms_h, float(mscale_spread),
                    float(self.mscale_alpha), float(self.mscale_beta), float(self.mscale_min),
                )
            else:
                dev = device or self._inv_freqs_h.device
                mscale_h = torch.full((self.axes_dim[1] // 2,), base_ms_h,
                                       device=dev, dtype=torch.float32)
        else:
            mscale_h = None

        # Width axis SEGA
        if ntk_factor_w > 1.0 and hasattr(self, '_inv_freqs_w'):
            base_ms_w = compute_base_mscale(
                target_res_w or self.training_resolution_w,
                self.training_resolution_w,
                self.base_mscale_formula,
                self.base_mscale_coefficient,
            )
            use_sega_w = (
                energy_profile_w is not None
                and float(mscale_spread) > 0.0
                and base_ms_w > 1.0 + 1e-8
            )
            if use_sega_w:
                inv_f = self._inv_freqs_w.to(device) if device is not None else self._inv_freqs_w
                mscale_w = compute_sega_allocation(
                    energy_profile_w, inv_f,
                    base_ms_w, float(mscale_spread),
                    float(self.mscale_alpha), float(self.mscale_beta), float(self.mscale_min),
                )
            else:
                dev = device or self._inv_freqs_w.device
                mscale_w = torch.full((self.axes_dim[2] // 2,), base_ms_w,
                                       device=dev, dtype=torch.float32)
        else:
            mscale_w = None

        if mscale_h is None and mscale_w is None:
            return None

        dev = device or (mscale_h.device if mscale_h is not None else mscale_w.device)
        if mscale_t is None:
            mscale_t = torch.ones(self.axes_dim[0] // 2, device=dev)
        if mscale_h is None:
            mscale_h = torch.ones(self.axes_dim[1] // 2, device=dev)
        if mscale_w is None:
            mscale_w = torch.ones(self.axes_dim[2] // 2, device=dev)

        mscale_full = torch.cat([
            mscale_t,
            mscale_h,
            mscale_w,
        ])
        # vals = [f"{v:.4f}" for v in mscale_full.tolist()]
        # print(
        #     f"[SEGA] mscale (per-dim, {mscale_full.shape[0]} dims): "
        #     f"min={mscale_full.min().item():.4f}  "
        #     f"mean={mscale_full.mean().item():.4f}  "
        #     f"max={mscale_full.max().item():.4f}\n"
        #     f"[SEGA] mscale values: {vals}"
        # )
        return mscale_full

    @functools.lru_cache(maxsize=None)
    def compute_condition_freqs(self, frame: int, height: int, width: int, device: torch.device, debug: int):
        seq_lens = frame * height * width
        pos_freqs_t = self.pos_freqs_t.to(device) if device is not None else self.pos_freqs_t
        pos_freqs_h = self.pos_freqs_h.to(device) if device is not None else self.pos_freqs_h
        pos_freqs_w = self.pos_freqs_w.to(device) if device is not None else self.pos_freqs_w
        neg_freqs_t = self.neg_freqs_t.to(device) if device is not None else self.neg_freqs_t
        neg_freqs_h = self.neg_freqs_h.to(device) if device is not None else self.neg_freqs_h
        neg_freqs_w = self.neg_freqs_w.to(device) if device is not None else self.neg_freqs_w

        freqs_frame = pos_freqs_t[:frame]
        freqs_height = pos_freqs_h[:height]
        freqs_width = pos_freqs_w[:width]

        freqs_frame = torch.stack([freqs_frame.cos(), -freqs_frame.sin(), freqs_frame.sin(), freqs_frame.cos()], dim=-1)
        freqs_height = torch.stack([freqs_height.cos(), -freqs_height.sin(), freqs_height.sin(), freqs_height.cos()], dim=-1)
        freqs_width = torch.stack([freqs_width.cos(), -freqs_width.sin(), freqs_width.sin(), freqs_width.cos()], dim=-1)

        em_T_H_W_D = torch.cat(
            [
                repeat(freqs_frame, "t d x -> t h w d x", h=height, w=width),
                repeat(freqs_height, "h d x -> t h w d x", t=frame, w=width),
                repeat(freqs_width, "w d x -> t h w d x", t=frame, h=height),
            ]
            , dim=-2,
        )
        return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()

    def forward(
        self, 
        x_B_T_H_W_D: torch.Tensor, 
        ntk_t: float = 1.0, 
        ntk_h: float = 1.0, 
        ntk_w: float = 1.0, 
        mscale_spread: Optional[float] = None,
        energy_profile_t: Optional[torch.Tensor] = None,
        energy_profile_h: Optional[torch.Tensor] = None,
        energy_profile_w: Optional[torch.Tensor] = None,
        target_resolution_h: Optional[int] = None,
        target_resolution_w: Optional[int] = None,
        debug: int = 0b0
        ) -> torch.Tensor:
        device = x_B_T_H_W_D.device
        _, frame, height, width, _ = x_B_T_H_W_D.shape
        self.ensure_ntk_factor(frame, height, width, ntk_t, ntk_h, ntk_w, debug)
        video_freq = self.compute_condition_freqs(frame, height, width, device, debug)
        mscale = self.compute_mscale(
            ntk_factor_t=ntk_t, 
            ntk_factor_h=ntk_h, 
            ntk_factor_w=ntk_w, 
            mscale_spread=mscale_spread,
            energy_profile_t=energy_profile_t, 
            energy_profile_h=energy_profile_h, 
            energy_profile_w=energy_profile_w, 
            target_res_h=target_resolution_h, 
            target_res_w=target_resolution_w, 
            device=device,
            debug=debug
            )
        if mscale is not None:
            return video_freq * mscale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        else:
            return video_freq

@torch.no_grad()
def compute_axis_spectral_profiles(hidden_states, height, width, n_bins_h, n_bins_w):
    reordered_hidden_states = rearrange(hidden_states, "b c t h w -> b (t h w) c")
    B, S, C = reordered_hidden_states.shape   # x_B_C_T_H_W
    n_spatial = min(S, height * width)
    spatial = reordered_hidden_states[:, :n_spatial].reshape(B, height, width, C)
    sm = spatial.float().mean(dim=(0, -1))
    sm = sm - sm.mean()

    def _axis_profile(sm, axis, n_bins, length):
        fft = torch.fft.fft(sm, dim=axis)
        power = fft.abs().pow(2).mean(dim=1 - axis)
        half = length // 2 + 1
        power = power[:half]
        freq_norm = torch.linspace(0.0, 1.0, half, device=power.device)
        bin_idx = (freq_norm * n_bins).long().clamp(0, n_bins - 1)
        energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_sum.scatter_add_(0, bin_idx, power.float())
        energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(power, dtype=torch.float32))
        return energy_sum / (energy_cnt + 1e-8)

    return (
        _axis_profile(sm, axis=0, n_bins=n_bins_h, length=height),
        _axis_profile(sm, axis=1, n_bins=n_bins_w, length=width),
    )

@torch.no_grad()
def compute_spectral_energy_profile(hidden_states, height, width, n_bins):
    reordered_hidden_states = rearrange(hidden_states, "b c t h w -> b (t h w) c")
    B, S, C = reordered_hidden_states.shape
    n_spatial = min(S, height * width)
    spatial = reordered_hidden_states[:, :n_spatial].reshape(B, height, width, C)
    spatial_map = spatial.float().mean(dim=(0, -1))
    spatial_map = spatial_map - spatial_map.mean()

    power = torch.fft.fftshift(torch.fft.fft2(spatial_map)).abs().pow(2)

    cy, cx = height / 2.0, width / 2.0
    y = torch.arange(height, device=power.device, dtype=torch.float32) - cy
    x = torch.arange(width, device=power.device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius_norm = torch.sqrt(yy ** 2 + xx ** 2)
    radius_norm = radius_norm / (radius_norm.max() + 1e-8)

    bin_idx = (radius_norm * n_bins).long().clamp(0, n_bins - 1).flatten()
    flat_pw = power.flatten()

    energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_sum.scatter_add_(0, bin_idx, flat_pw)
    energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(flat_pw))
    return energy_sum / (energy_cnt + 1e-8)

@torch.no_grad()
def compute_dynamic_spread(energy_profile, spread_min=0.0, spread_max=1.0, alpha=1.5):
    eps = 1e-8
    energy = energy_profile.clamp(min=eps)
    geo_mean = torch.exp(torch.log(energy).mean())
    arith_mean = energy.mean()
    flatness = (geo_mean / (arith_mean + eps)).clamp(0.0, 1.0)
    concentration = 1.0 - flatness.item()
    return spread_min + (spread_max - spread_min) * (1.0 - (1.0 - concentration) ** alpha)

def prepare_embedded_sequence(
    class_obj: MiniTrainDIT,
    x_B_C_T_H_W: torch.Tensor,
    fps: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    transformer_options: Optional[dict] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

    Args:
        x_B_C_T_H_W (torch.Tensor): video
        fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                If None, a default value (`class_obj.base_fps`) will be used.
        padding_mask (Optional[torch.Tensor]): current it is not used

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - A tensor of shape (B, T, H, W, D) with the embedded sequence.
            - An optional positional embedding tensor, returned only if the positional embedding class
            (`class_obj.pos_emb_cls`) includes 'rope'. Otherwise, None.

    Notes:
        - If `class_obj.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
        - The method of applying positional embeddings depends on the value of `class_obj.pos_emb_cls`.
        - If 'rope' is in `class_obj.pos_emb_cls` (case insensitive), the positional embeddings are generated using
            the `class_obj.pos_embedder` with the shape [T, H, W].
        - If "fps_aware" is in `class_obj.pos_emb_cls`, the positional embeddings are generated using the
        `class_obj.pos_embedder` with the fps tensor.
        - Otherwise, the positional embeddings are generated without considering fps.
    """
    if class_obj.concat_padding_mask:
        if padding_mask is None:
            padding_mask = torch.zeros(x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4], dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
        else:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
        x_B_C_T_H_W = torch.cat(
            [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
        )
    x_B_T_H_W_D = class_obj.x_embedder(x_B_C_T_H_W)

    if class_obj.extra_per_block_abs_pos_emb:
        extra_pos_emb = class_obj.extra_pos_embedder(x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype)
    else:
        extra_pos_emb = None

    if "rope" in class_obj.pos_emb_cls.lower():
        state: dict = transformer_options.get(File_x_SEGA_Anima_state, None)
        if state is None:
            raise RuntimeError("Failed to pass parameters to internal function.")
        training_resolution_h: int = state.get("training_resolution_h", 1024)
        training_resolution_w: int = state.get("training_resolution_w", 1024)
        debug: int = state.get("debug", 0)
        d_mul: float = state.get("debug", 0)
        log_s_mul: float = state.get("debug", 0)
        axes_dims_rope: tuple[int, int, int] = state.get("axes_dims_rope", (44, 42, 42))
        rope_embedder: File_x_SEGA_Anima_RoPE = state.get("RoPE_Embedder", None)
        
        if rope_embedder is None:
            raise RuntimeError("Class File_x_SEGA_Anima_RoPE is not properly transfered.")

        img_h = x_B_C_T_H_W.shape[3]
        img_w = x_B_C_T_H_W.shape[4]
        target_resolution_h = img_h * 8
        target_resolution_w = img_w * 8
        s_h = target_resolution_h / training_resolution_h
        s_w = target_resolution_w / training_resolution_w
        d_h = axes_dims_rope[1]
        d_w = axes_dims_rope[2]
        ntk_h = s_h ** (d_mul * d_h / (d_h - 2)) / (1 + log_s_mul * math.log(s_h))
        ntk_w = s_w ** (d_mul * d_w / (d_w - 2)) / (1 + log_s_mul * math.log(s_w))
        big = max(float(s_h), float(s_w))
        small = min(float(s_h), float(s_w))
        if big > 1.0 and small > 0.0 and abs(s_h - s_w) > 1e-8:
            ntk_big = max(float(ntk_h), float(ntk_w))
            ntk_small = 1.0 + (small / big) * max(0.0, (ntk_big - 1.0))
            if s_h < s_w:
                ntk_h = ntk_small
                ntk_w = ntk_big
            else:
                ntk_h = ntk_big
                ntk_w = ntk_small

        if max(ntk_h, ntk_w) > 1.0 and img_h > 0 and img_w > 0:
            n_bins_div = 1 if debug & 0b100 == 0 else 2
            n_bins_h = max(img_h // n_bins_div, 8)
            n_bins_w = max(img_w // n_bins_div, 8)
            energy_profile_h, energy_profile_w = compute_axis_spectral_profiles(x_B_C_T_H_W, img_h, img_w, n_bins_h, n_bins_w)
            iso_profile = compute_spectral_energy_profile(x_B_C_T_H_W, img_h, img_w, max(img_h, img_w) // 2)
            dynamic_spread = compute_dynamic_spread(iso_profile, 0.0, 1.0, 1.5)
        else:
            energy_profile_h = None
            energy_profile_w = None
            dynamic_spread = None

        rope_emb_L_1_1_D = rope_embedder(
            x_B_T_H_W_D, 
            ntk_h=ntk_h, 
            ntk_w=ntk_w, 
            mscale_spread=dynamic_spread, 
            energy_profile_h=energy_profile_h, 
            energy_profile_w=energy_profile_w, 
            target_resolution_h=target_resolution_h, 
            target_resolution_w=target_resolution_w,
            debug=debug
            )
        return x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb
    x_B_T_H_W_D = x_B_T_H_W_D + class_obj.pos_embedder(x_B_T_H_W_D, device=x_B_C_T_H_W.device)  # [B, T, H, W, D]

    return x_B_T_H_W_D, None, extra_pos_emb

def File_x_SEGA_Anima_wrapper_forward(
        class_obj: MiniTrainDIT,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
    orig_shape = list(x.shape)
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (class_obj.patch_temporal, class_obj.patch_spatial, class_obj.patch_spatial))
    x_B_C_T_H_W = x
    timesteps_B_T = timesteps
    crossattn_emb = context
    transformer_options: dict = kwargs.get("transformer_options", {})
    """
    Args:
        x: (B, C, T, H, W) tensor of spatial-temp inputs
        timesteps: (B, ) tensor of timesteps
        crossattn_emb: (B, N, D) tensor of cross-attention embeddings
    """
    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = prepare_embedded_sequence(
        class_obj,
        x_B_C_T_H_W,
        fps=fps,
        padding_mask=padding_mask,
        transformer_options=transformer_options
    )

    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)
    t_embedding_B_T_D, adaln_lora_B_T_3D = class_obj.t_embedder[1](class_obj.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype))
    t_embedding_B_T_D = class_obj.t_embedding_norm(t_embedding_B_T_D)

    # for logging purpose
    affline_scale_log_info = {}
    affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
    class_obj.affline_scale_log_info = affline_scale_log_info
    class_obj.affline_emb = t_embedding_B_T_D
    class_obj.crossattn_emb = crossattn_emb

    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        assert (
            x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
        ), f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        "transformer_options": transformer_options,
    }

    # The residual stream for this model has large values. To make fp16 compute_dtype work, we keep the residual stream
    # in fp32, but run attention and MLP modules in fp16.
    # An alternate method that clamps fp16 values "works" in the sense that it makes coherent images, but there is noticeable
    # quality degradation and visual artifacts.
    if x_B_T_H_W_D.dtype == torch.float16:
        x_B_T_H_W_D = x_B_T_H_W_D.float()

    for block in class_obj.blocks:
        x_B_T_H_W_D = block(
            x_B_T_H_W_D,
            t_embedding_B_T_D,
            crossattn_emb,
            **block_kwargs,
        )

    x_B_T_H_W_O = class_obj.final_layer(x_B_T_H_W_D.to(crossattn_emb.dtype), t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
    x_B_C_Tt_Hp_Wp = class_obj.unpatchify(x_B_T_H_W_O)[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]
    return x_B_C_Tt_Hp_Wp

def File_x_SEGA_Anima_wrapper(executor: WrapperExecutor, x, timesteps, context, fps, padding_mask, **kwargs):
    return File_x_SEGA_Anima_wrapper_forward(executor.class_obj, x, timesteps, context, fps, padding_mask, **kwargs)

class File_x_SEGA_Anima_(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="File_x SEGA Anima",
                         display_name="SEGA Anima",
                         category="SEGA/Anima",
                         inputs=[io.Model.Input(id="model"),
                                 io.Int.Input(id="training_resolution", default=1024, min=1, max=65536, step=1),
                                 io.Float.Input(id="theta", default=10000.0, min=-100000.0, max=100000.0, step=0.001),
                                 io.Combo.Input(id="base_mscale_formula", options=["power_res", "log_res"], default="power_res"),
                                 io.Float.Input(id="base_mscale_coefficient", default=0.08, min=-100.000, max=100.000, step=0.001),
                                 io.Float.Input(id="mscale_alpha", default=0.15, min=-100.000, max=100.000, step=0.001),
                                 io.Float.Input(id="mscale_beta", default=1.5, min=-100.000, max=100.000, step=0.001),
                                 io.Float.Input(id="mscale_min", default=1.0, min=-100.000, max=100.000, step=0.001),
                                 io.Float.Input(id="d_mul", default=2.0, min=-100.000, max=100.000, step=0.001),
                                 io.Float.Input(id="log_s_mul", default=0.1, min=-100.000, max=100.000, step=0.001),
                                 # io.Int.Input(id="debug", default=0b0, min=0, max=0b1111_1111_1111_1111, step=1, advanced=True),
                                 ],
                         outputs=[io.Model.Output(id=None),],)
    
    @classmethod
    def execute(cls, model: ModelPatcher, training_resolution, theta, base_mscale_formula, base_mscale_coefficient, 
                mscale_alpha, mscale_beta, mscale_min, d_mul, log_s_mul):
        base_model = getattr(model, "model", None)
        if base_model is None:
            raise RuntimeError("Model input has no model object.")
        image_model = None
        model_config = getattr(base_model, "model_config", None)
        if model_config is not None:
            image_model = model_config.unet_config.get("image_model", None)
        if image_model != "anima":
            raise RuntimeError("Model input is not Anima model.")
        diffusion_model = getattr(base_model, "diffusion_model", None)
        if diffusion_model is None:
            raise RuntimeError("Model input has invalid Anima model.")
        modified_model = model.clone()
        head_dim = modified_model.model.diffusion_model.model_channels // modified_model.model.diffusion_model.num_heads
        dim_h = head_dim // 6 * 2
        dim_w = dim_h
        dim_t = head_dim - 2 * dim_h
        assert head_dim == dim_h + dim_w + dim_t, f"bad dim: {head_dim} != {dim_h} + {dim_w} + {dim_t}"
        axes_dims_rope = (dim_t, dim_h, dim_w)
        modified_model.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, File_x_SEGA_Anima_key)
        transformer_options = modified_model.model_options.setdefault("transformer_options", {})
        transformer_options[File_x_SEGA_Anima_state] = {
            "training_resolution_h": training_resolution,
            "training_resolution_w": training_resolution,
            "debug": 0b0,
            "d_mul": d_mul,
            "log_s_mul": log_s_mul,
            "axes_dims_rope": axes_dims_rope,
            "RoPE_Embedder": File_x_SEGA_Anima_RoPE(axes_dims_rope, theta, base_mscale_formula, base_mscale_coefficient, 
                                                    training_resolution, training_resolution, mscale_alpha, mscale_beta, mscale_min, 0b0)
        }
        modified_model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, File_x_SEGA_Anima_key, File_x_SEGA_Anima_wrapper)
        return io.NodeOutput(modified_model)
