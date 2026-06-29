# SEGA Anima

SEGA anima is implementation of ["SEGA: Spectral-Energy Guided Attention for Resolution Extrapolation in Diffusion Transformers" (arXiv:2605.22668)](https://arxiv.org/abs/2605.22668) for Anima model in Comfyui.

## What should I do when output image is bad

1. If image does not appear, first thing you have to do is decreasing "training_resolution". Too low "training_resolution" results structual inaccuracy.
2. Adjust CFG. Normally, higher CFG is needed for high resolution output.
3. You may adjust "theta", "base_mscale_coefficient", "mscale_alpha" or "d_mul".
4. Creator of the this node does not know what rest of parameters do. You may do experience with those.
5. If changing none of above work, Increasing steps may work.