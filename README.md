# SEGA Anima

SEGA anima is implementation of ["SEGA: Spectral-Energy Guided Attention for Resolution Extrapolation in Diffusion Transformers" (arXiv:2605.22668)](https://arxiv.org/abs/2605.22668) for Anima model in Comfyui.

## What should I do when output image is bad

1. First thing you should to do is increasing CFG. If you could not find good CFG, come back later.
2. Reduce "training_resolution" if image does not appear. Too low "training_resolution" results structual inaccuracy. In that case, increase "training_resolution" or go to next step.
3. You may increase "theta" or "base_mscale_coefficient" or "d_mul".
4. Creator of the this node does not know what rest of parameters do. You may do experience with those.