# SEGA Anima

SEGA anima is implementation of ["SEGA: Spectral-Energy Guided Attention for Resolution Extrapolation in Diffusion Transformers" (arXiv:2605.22668)](https://arxiv.org/abs/2605.22668) for Anima model in Comfyui.

## What should I do when output image is bad

- Decrease "training_resolution" if image does not appear. Too low "training_resolution" results structual inaccuracy.
- Adjust CFG. Normally, higher CFG is needed for high resolution output.
- You may increase "theta" or "base_mscale_coefficient" or "d_mul".
- You may decrease "mscale_alpha".
- Creator of the this node does not know what rest of parameters do. You may do experience with those.
- If changing none of above work, Increasing steps may work.