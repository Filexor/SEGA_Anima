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
        if debug & 0b1000_0000 == 0:
            self.build_freqs(1.0, 1.0, 1.0, debug)
        else:
            self.build_freqs_with_size(self.frame, self.height, self.width, 1.0, 1.0, 1.0, debug)

    @staticmethod
    def rope_params(index, dim, theta=10000, ntk_factor=1.0, debug=0b0):
        assert dim % 2 == 0
        scaled_theta = theta * ntk_factor
        step = 2 if debug & 0b1_0000 == 0 else 1
        freqs = torch.outer(index, 1.0 / torch.pow(scaled_theta, torch.arange(0, dim, step).to(torch.float32).div(dim)))
        if debug & 0b10_0000 == 0:
            freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def build_freqs(self, ntk_t: float, ntk_h: float, ntk_w: float, debug:int):
        """(Re)build pos/neg frequency tensors for the given per-axis NTK factors."""
        pos_index = torch.arange(max(self.axes_dim), dtype=torch.float)
        neg_index = torch.arange(max(self.axes_dim), dtype=torch.float).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta, ntk_factor=ntk_t, debug=debug),
                self.rope_params(pos_index, self.axes_dim[1], self.theta, ntk_factor=ntk_h, debug=debug),
                self.rope_params(pos_index, self.axes_dim[2], self.theta, ntk_factor=ntk_w, debug=debug),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta, ntk_factor=ntk_t, debug=debug),
                self.rope_params(neg_index, self.axes_dim[1], self.theta, ntk_factor=ntk_h, debug=debug),
                self.rope_params(neg_index, self.axes_dim[2], self.theta, ntk_factor=ntk_w, debug=debug),
            ],
            dim=1,
        )

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

    def build_freqs_with_size(self, frame, height, width, ntk_t: float, ntk_h: float, ntk_w: float, debug:int):
        """(Re)build pos/neg frequency tensors for the given per-axis NTK factors."""
        pos_index_frame = torch.arange(frame, dtype=torch.float)
        pos_index_height = torch.arange(height, dtype=torch.float)
        pos_index_width = torch.arange(width, dtype=torch.float)
        neg_index_frame = torch.arange(frame, dtype=torch.float).flip(0) * -1 - 1
        neg_index_height = torch.arange(height, dtype=torch.float).flip(0) * -1 - 1
        neg_index_width = torch.arange(width, dtype=torch.float).flip(0) * -1 - 1

        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index_frame, self.axes_dim[0], self.theta, ntk_factor=ntk_t, debug=debug),
                self.rope_params(pos_index_height, self.axes_dim[1], self.theta, ntk_factor=ntk_h, debug=debug),
                self.rope_params(pos_index_width, self.axes_dim[2], self.theta, ntk_factor=ntk_w, debug=debug),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index_frame, self.axes_dim[0], self.theta, ntk_factor=ntk_t, debug=debug),
                self.rope_params(neg_index_height, self.axes_dim[1], self.theta, ntk_factor=ntk_h, debug=debug),
                self.rope_params(neg_index_width, self.axes_dim[2], self.theta, ntk_factor=ntk_w, debug=debug),
            ],
            dim=1,
        )

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
            if debug & 0x1000_0000 == 0:
                self.build_freqs(ntk_t, ntk_h, ntk_w, debug)
            else:
                self.build_freqs_with_size(frame, height, width, ntk_t, ntk_h, ntk_w, debug)

    @torch.no_grad()
    def compute_mscale(self, ntk_factor_t, ntk_factor_h, ntk_factor_w, mscale_spread,
                       t_energy_profile=None, energy_profile_h=None, energy_profile_w=None,
                       target_res_h=None, target_res_w=None, device=None):
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
        pos_freqs = self.pos_freqs.to(device) if device is not None else self.pos_freqs
        neg_freqs = self.neg_freqs.to(device) if device is not None else self.neg_freqs

        step = 2 if debug & 0b1_0000 == 0 else 1
        freqs_pos = pos_freqs.split([x // step for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // step for x in self.axes_dim], dim=1)

        if debug & 0b100_0000 == 0:
            # freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
            if debug & 0b1000 != 0:
                freqs_frame = torch.cat([freqs_neg[0][-(frame - frame // 2) :], freqs_pos[0][: frame // 2]], dim=0)
                freqs_frame = freqs_frame.view(frame, 1, 1, -1).expand(frame, height, width, -1)
                freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
                freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
            else:
                freqs_frame = freqs_pos[0][:frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

            freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
            return freqs.clone().contiguous()
        else:
            if debug & 0b1000 != 0:
                freqs_frame = torch.cat([freqs_neg[0][-(frame - frame // 2) :], freqs_pos[0][: frame // 2]], dim=0)
                freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
                freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            else:
                freqs_frame = freqs_pos[0][:frame]
                freqs_height = freqs_pos[1][:height]
                freqs_width = freqs_pos[2][:width]

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
        energy_profile_h: Optional[torch.Tensor] = None,
        energy_profile_w: Optional[torch.Tensor] = None,
        energy_profile_t: Optional[torch.Tensor] = None,
        target_resolution_h: Optional[int] = None,
        target_resolution_w: Optional[int] = None,
        debug: int = 0b0
        ) -> torch.Tensor:
        device = x_B_T_H_W_D.device
        _, frame, height, width, _ = x_B_T_H_W_D.shape
        self.ensure_ntk_factor(frame, height, width, ntk_t, ntk_h, ntk_w, debug)
        video_freq = self.compute_condition_freqs(frame, height, width, device, debug)
        mscale = self.compute_mscale(ntk_t, ntk_h, ntk_w, mscale_spread, energy_profile_h, energy_profile_w, energy_profile_t, target_resolution_h, target_resolution_w, device)
        if mscale is not None:
            if debug & 0b100_0000 == 0:
                return video_freq * mscale
            else:
                return video_freq * mscale.unsqueeze(-1)
        else:
            return video_freq

def compute_effective_mscale(
    dim_range: torch.Tensor,
    ntk_factor: float = 1.0,
    target_res: Optional[int] = None, 
    training_res: Optional[int] = None, 
    base_mscale_formula: str = "power_res",
    base_mscale_coefficient: Optional[float] = None,
    energy_profile: Optional[torch.Tensor] = None,
    mscale_spread: Optional[float] = None,
    mscale_alpha: float = 0.15,
    mscale_beta: float = 1.5,
    mscale_min: float = 1.0,
) -> tuple[Union[float, torch.Tensor], torch.Tensor]: 
    
    theta = 10000.0
    inv_freq = 1.0 / (theta ** dim_range)
    effective_mscale: Union[float, torch.Tensor] = 1.0

    if ntk_factor > 1.0:
        scaled_theta = theta * ntk_factor
        freqs = 1.0 / (scaled_theta ** dim_range)

        base_ms = compute_base_mscale(
            target_res=target_res,
            training_res=training_res,
            formula=base_mscale_formula,
            coefficient=base_mscale_coefficient,
        )

        use_sega = (
            energy_profile is not None
            and mscale_spread is not None
            and float(mscale_spread) > 0.0
            and base_ms > 1.0 + 1e-8
        )

        if use_sega:
            return compute_sega_allocation(
                energy_profile=energy_profile,
                freqs=freqs,
                base_mscale=base_ms,
                spread=float(mscale_spread),
                alpha=float(mscale_alpha),
                beta=float(mscale_beta),
                min_mscale=float(mscale_min),
            ), freqs
        else:
            return base_ms, freqs
    else:
        return effective_mscale, inv_freq

def generate_embeddings_old(
    class_obj: VideoRopePosition3DEmb,
    B_T_H_W_C: torch.Tensor,
    fps: Optional[torch.Tensor] = None,
    h_ntk_factor: Optional[float] = None,
    w_ntk_factor: Optional[float] = None,
    t_ntk_factor: Optional[float] = None,
    device=None,
    dtype=None,
    target_res: Optional[int] = None, 
    training_res: Optional[int] = None, 
    base_mscale_formula: str = "power_res",
    base_mscale_coefficient: Optional[float] = None,
    h_energy_profile: Optional[torch.Tensor] = None,
    w_energy_profile: Optional[torch.Tensor] = None,
    t_energy_profile: Optional[torch.Tensor] = None,
    mscale_spread: Optional[float] = None,
    mscale_alpha: float = 0.15,
    mscale_beta: float = 1.5,
    mscale_min: float = 1.0,
    debug: int = 0b0,
):
    """
    Generate embeddings for the given input size.

    Args:
        B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
        fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
        h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses class_obj.h_ntk_factor.
        w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses class_obj.w_ntk_factor.
        t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses class_obj.t_ntk_factor.

    Returns:
        Not specified in the original code snippet.
    """
    # this is where incorrect: ntk_factors are normally None.
    h_ntk_factor = None if debug & 0b1_0000_0000 else h_ntk_factor
    w_ntk_factor = None if debug & 0b10_0000_0000 else w_ntk_factor
    t_ntk_factor = None if debug & 0b100_0000_0000 else t_ntk_factor
    h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else class_obj.h_ntk_factor
    w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else class_obj.w_ntk_factor
    t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else class_obj.t_ntk_factor

    h_effective_mscale, h_spatial_freqs = compute_effective_mscale(class_obj.dim_spatial_range.to(device=device), h_ntk_factor, target_res, training_res, base_mscale_formula, base_mscale_coefficient, h_energy_profile, mscale_spread, mscale_alpha, mscale_beta, mscale_min)
    w_effective_mscale, w_spatial_freqs = compute_effective_mscale(class_obj.dim_spatial_range.to(device=device), w_ntk_factor, target_res, training_res, base_mscale_formula, base_mscale_coefficient, w_energy_profile, mscale_spread, mscale_alpha, mscale_beta, mscale_min)
    t_effective_mscale, temporal_freqs = compute_effective_mscale(class_obj.dim_temporal_range.to(device=device), t_ntk_factor, target_res, training_res, base_mscale_formula, base_mscale_coefficient, t_energy_profile, mscale_spread, mscale_alpha, mscale_beta, mscale_min)

    B, T, H, W, _ = B_T_H_W_C.shape
    seq = torch.arange(max(H, W, T), dtype=torch.float, device=device)
    uniform_fps = (fps is None) or isinstance(fps, (int, float)) or (fps.min() == fps.max())
    assert (
        uniform_fps or B == 1 or T == 1
    ), "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
    half_emb_h = torch.outer(seq[:H].to(device=device), h_spatial_freqs)
    half_emb_w = torch.outer(seq[:W].to(device=device), w_spatial_freqs)

    # apply sequence scaling in temporal dimension
    if fps is None or class_obj.enable_fps_modulation is False:  # image case
        half_emb_t = torch.outer(seq[:T].to(device=device), temporal_freqs)
    else:
        half_emb_t = torch.outer(seq[:T].to(device=device) / fps * class_obj.base_fps, temporal_freqs)

    half_emb_h = torch.stack([torch.cos(half_emb_h), -torch.sin(half_emb_h), torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1)
    half_emb_w = torch.stack([torch.cos(half_emb_w), -torch.sin(half_emb_w), torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1)
    half_emb_t = torch.stack([torch.cos(half_emb_t), -torch.sin(half_emb_t), torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1)

    if debug & 0b10:
        if isinstance(h_effective_mscale, torch.Tensor):
            half_emb_h *= h_effective_mscale.unsqueeze(0).unsqueeze(2).repeat(1, 1, 4)
        else:
            half_emb_h *= h_effective_mscale
    if debug & 0b100:
        if isinstance(w_effective_mscale, torch.Tensor):
            half_emb_w *= w_effective_mscale.unsqueeze(0).unsqueeze(2).repeat(1, 1, 4)
        else:
            half_emb_w *= w_effective_mscale
    if debug & 0b1000:
        if isinstance(t_effective_mscale, torch.Tensor):
            half_emb_t *= t_effective_mscale.unsqueeze(0).unsqueeze(2).repeat(1, 1, 4)
        else:
            half_emb_t *= t_effective_mscale

    em_T_H_W_D = torch.cat(
        [
            repeat(half_emb_t, "t d x -> t h w d x", h=H, w=W),
            repeat(half_emb_h, "h d x -> t h w d x", t=T, w=W),
            repeat(half_emb_w, "w d x -> t h w d x", t=T, h=H),
        ]
        , dim=-2,
    )

    return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()    # rope_emb_L_1_1_D

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

def prepare_embedded_sequence_old(
    class_obj: MiniTrainDIT,
    x_B_C_T_H_W: torch.Tensor,
    fps: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    h_ntk_factor: Optional[float] = None,
    w_ntk_factor: Optional[float] = None,
    t_ntk_factor: Optional[float] = None,
    target_res: Optional[int] = None, 
    training_res: Optional[int] = None, 
    base_mscale_formula: str = "power_res",
    base_mscale_coefficient: Optional[float] = None,
    h_energy_profile: Optional[torch.Tensor] = None,
    w_energy_profile: Optional[torch.Tensor] = None,
    t_energy_profile: Optional[torch.Tensor] = None,
    mscale_spread: Optional[float] = None,
    mscale_alpha: float = 0.15,
    mscale_beta: float = 1.5,
    mscale_min: float = 1.0,
    debug: int = 0b0,
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

    # inside of this is likely wrong
    if "rope" in class_obj.pos_emb_cls.lower():
        rope_emb_L_1_1_D = generate_embeddings(
            class_obj = class_obj.pos_embedder,
            B_T_H_W_C = x_B_T_H_W_D,
            fps = fps,
            h_ntk_factor = h_ntk_factor,
            w_ntk_factor = w_ntk_factor, 
            t_ntk_factor = t_ntk_factor, 
            device = x_B_C_T_H_W.device,
            target_res = target_res,
            training_res = training_res,
            base_mscale_formula = base_mscale_formula,
            base_mscale_coefficient = base_mscale_coefficient,
            h_energy_profile = h_energy_profile,
            w_energy_profile = w_energy_profile,
            t_energy_profile = t_energy_profile,
            mscale_spread = mscale_spread,
            mscale_alpha = mscale_alpha,
            mscale_beta = mscale_beta,
            mscale_min = mscale_min,
            debug = debug,
            )
        return x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb
    x_B_T_H_W_D = x_B_T_H_W_D + class_obj.pos_embedder(x_B_T_H_W_D, device=x_B_C_T_H_W.device)  # [B, T, H, W, D]

    return x_B_T_H_W_D, None, extra_pos_emb

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
        d_mul = 2 if debug & 0b1 == 0 else 1
        log_s_mul = 1.0 if debug & 0b10 == 0 else 0.1
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

def File_x_SEGA_Anima_wrapper_forward_old(
    class_obj: MiniTrainDIT,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    fps: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    **kwargs,
    ):

    orig_shape = list(x.shape)  # [2, 16, 1, 114, 114] at 912 * 912
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (class_obj.patch_temporal, class_obj.patch_spatial, class_obj.patch_spatial))
    x_B_C_T_H_W = x # [2, 16, 1, 114, 114] at 912 * 912
    timesteps_B_T = timesteps
    crossattn_emb = context
    transformer_options: dict = kwargs.get("transformer_options", {})
    """
    Args:
        x: (B, C, T, H, W) tensor of spatial-temp inputs
        timesteps: (B, ) tensor of timesteps
        crossattn_emb: (B, N, D) tensor of cross-attention embeddings
    """

    state: dict = transformer_options.get(File_x_SEGA_Anima_state, None)
    if state is None:
        raise RuntimeError("Failed to pass parameters to internal function.")
    training_resolution = state.get("training_resolution", 1024)
    base_mscale_formula = state.get("base_mscale_formula", "power_res")
    base_mscale_coefficient = state.get("base_mscale_coefficient", 0.08)
    mscale_alpha = state.get("mscale_alpha", 0.15)
    mscale_beta = state.get("mscale_beta", 1.5)
    mscale_min = state.get("mscale_min", 1.0)
    debug = state.get("debug", 0b0)

    patch_size = 2 if debug & 0b1_0000 else 1
    max_h = x.shape[-2]
    max_w = x.shape[-1]
    target_resolution = x_B_C_T_H_W.shape[-2] * patch_size* 8 * x_B_C_T_H_W.shape[-1] * patch_size* 8
    s = target_resolution / (training_resolution ** 2)
    head_dim = class_obj.model_channels // class_obj.num_heads
    dim_h = head_dim // 6 * 2
    dim_w = dim_h
    dim_t = head_dim - 2 * dim_h
    assert head_dim == dim_h + dim_w + dim_t, f"bad dim: {head_dim} != {dim_h} + {dim_w} + {dim_t}"
    axes_dims_rope = (dim_t, dim_h, dim_w)
    rope_spatial_dim = axes_dims_rope[1]
    d = rope_spatial_dim
    d_multiplier = 1 if debug & 0b10_0000 else 2
    if debug & 0b1: # True is correct
        current_ntk_factor = s ** (d_multiplier * d / (d - 2)) / (1 + 0.1 * math.log(s))
    else:
        current_ntk_factor = s ** (d_multiplier * d / (d - 2)) / (1 + math.log(s))

    dynamic_spread = None
    energy_profile_h = None
    energy_profile_w = None
    energy_profile_t = None

    divider = 2 if debug & 0b100_0000 else 1
    n_bins_divider = 1 if debug & 0b1000_0000 else 2

    if current_ntk_factor > 1.0:
        n_spatial = x.shape[-2] * x.shape[-1]
        if max_h * max_w == n_spatial:
            img_h, img_w = max_h, max_w
        else:
            aspect = max_h / max_w
            img_w = int(math.sqrt(n_spatial / aspect))
            img_h = n_spatial // max(1, img_w)

        n_bins_h = max(img_h // divider, 8)
        n_bins_w = max(img_w // divider, 8)
        energy_profile_h, energy_profile_w = compute_axis_spectral_profiles(
            x, img_h, img_w, n_bins_h, n_bins_w,
        )
        iso_profile = compute_spectral_energy_profile(
            x, img_h, img_w, n_bins=max(img_h, img_w) // n_bins_divider,
        )
        dynamic_spread = compute_dynamic_spread(
            iso_profile,
            spread_min=0.0,  # class_obj.spread_min
            spread_max=1.0,  # class_obj.spread_max
            alpha=1.5,  # class_obj.spread_alpha
        )

    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = prepare_embedded_sequence(
        class_obj = class_obj,
        x_B_C_T_H_W = x_B_C_T_H_W,
        fps = fps,
        padding_mask = padding_mask,
        h_ntk_factor = current_ntk_factor,
        w_ntk_factor = current_ntk_factor, 
        t_ntk_factor = current_ntk_factor, 
        target_res = target_resolution,
        training_res = training_resolution,
        base_mscale_formula = base_mscale_formula,
        base_mscale_coefficient = base_mscale_coefficient,
        h_energy_profile = energy_profile_h,
        w_energy_profile = energy_profile_w,
        t_energy_profile = energy_profile_t,
        mscale_spread = dynamic_spread,
        mscale_alpha = mscale_alpha,
        mscale_beta = mscale_beta,
        mscale_min = mscale_min,
        debug = debug,
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
        "transformer_options": kwargs.get("transformer_options", {}),
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
                         category="File_x/SEGA",
                         inputs=[io.Model.Input(id="model"),
                                 io.Int.Input(id="training_resolution_h", default=1024, min=16, max=65536, step=8),
                                 io.Int.Input(id="training_resolution_w", default=1024, min=16, max=65536, step=8),
                                 io.Float.Input(id="theta", default=10000.0, min=-0, max=100000.0, step=0.001),
                                 io.Combo.Input(id="base_mscale_formula", options=["power_res", "log_res"], default="power_res"),
                                 io.Float.Input(id="base_mscale_coefficient", default=0.08, min=-100.000, max=100.000, step=0.001),
                                 io.Float.Input(id="mscale_alpha", default=0.15, min=-100.000, max=100.000, step=0.001),
                                 io.Float.Input(id="mscale_beta", default=1.5, min=-100.000, max=100.000, step=0.001),
                                 io.Float.Input(id="mscale_min", default=1.0, min=-100.000, max=100.000, step=0.001),
                                 io.Int.Input(id="debug", default=0b0, min=0, max=0b111_1111_1111, step=1, advanced=True),],
                         outputs=[io.Model.Output(id=None),],)
    
    @classmethod
    def execute(cls, model: ModelPatcher, training_resolution_h, training_resolution_w, theta, base_mscale_formula, base_mscale_coefficient, 
                mscale_alpha, mscale_beta, mscale_min, debug):
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
            "training_resolution_h": training_resolution_h,
            "training_resolution_w": training_resolution_w,
            "debug": debug,
            "axes_dims_rope": axes_dims_rope,
            "RoPE_Embedder": File_x_SEGA_Anima_RoPE(axes_dims_rope, theta, base_mscale_formula, base_mscale_coefficient, 
                                                    training_resolution_h, training_resolution_w, mscale_alpha, mscale_beta, mscale_min, debug)
        }
        modified_model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, File_x_SEGA_Anima_key, File_x_SEGA_Anima_wrapper)
        return io.NodeOutput(modified_model)
