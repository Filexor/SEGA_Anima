import comfy
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrapperExecutor, WrappersMP
from comfy_api.latest import io

from comfy.ldm.cosmos.predict2 import MiniTrainDIT
from comfy.ldm.cosmos.position_embedding import VideoRopePosition3DEmb

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

def generate_embeddings(
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

def prepare_embedded_sequence(
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

def File_x_SEGA_Anima_wrapper_forward(
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

def File_x_SEGA_Anima_wrapper(executor: WrapperExecutor, x, timesteps, context, fps, padding_mask, **kwargs):
    return File_x_SEGA_Anima_wrapper_forward(executor.class_obj, x, timesteps, context, fps, padding_mask, **kwargs)

class File_x_SEGA_Anima_(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="File_x SEGA Anima",
                         display_name="SEGA Anima",
                         category="File_x/SEGA",
                         inputs=[io.Model.Input(id="model"),
                                 io.Int.Input(id="training_resolution", default=1024, min=16, max=65536, step=8),
                                 io.Combo.Input(id="base_mscale_formula", options=["power_res", "log_res"], default="power_res"),
                                 io.Float.Input(id="base_mscale_coefficient", default=0.08, min=-10.000, max=10.000, step=0.001),
                                 io.Float.Input(id="mscale_alpha", default=0.15, min=-10.000, max=10.000, step=0.001),
                                 io.Float.Input(id="mscale_beta", default=1.5, min=-10.000, max=10.000, step=0.001),
                                 io.Float.Input(id="mscale_min", default=1.0, min=-10.000, max=10.000, step=0.001),
                                 io.Int.Input(id="debug", default=0b0, min=0, max=0b111_1111_1111, step=1, advanced=True),],
                         outputs=[io.Model.Output(id=None),],)
    
    @classmethod
    def execute(cls, model: ModelPatcher, training_resolution, base_mscale_formula, base_mscale_coefficient, mscale_alpha, mscale_beta, mscale_min, debug):
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
        modified_model.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, File_x_SEGA_Anima_key)
        transformer_options = modified_model.model_options.setdefault("transformer_options", {})
        transformer_options[File_x_SEGA_Anima_state] = {
            "training_resolution": training_resolution,
            "base_mscale_formula": base_mscale_formula,
            "base_mscale_coefficient": base_mscale_coefficient,
            "mscale_alpha": mscale_alpha,
            "mscale_beta": mscale_beta,
            "mscale_min": mscale_min,
            "debug": debug,
        }
        modified_model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, File_x_SEGA_Anima_key, File_x_SEGA_Anima_wrapper)
        return io.NodeOutput(modified_model)
