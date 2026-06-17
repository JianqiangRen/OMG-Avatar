# OMG-Avatar: One-shot Multi-LOD Gaussian Head Avatar


[![CVPR 2026](https://img.shields.io/badge/project-page-blue)](https://human3daigc.github.io/OMGAvatar_project_page/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.01506-b31b1b.svg)](https://arxiv.org/abs/2603.01506)

This repository contains the codes for **OMG-Avatar**, accepted to **CVPR 2026**.

## Overview

We propose a unified model that accommodates diverse hardware capabilities and inference-speed requirements through LOD head-avatar modeling. The key ingredients of our method are:

- A **Transformer-based architecture** for global feature extraction
- **Projection-based sampling** for local feature acquisition
- **Depth-buffer-guided fusion** ensuring occlusion plausibility
- A **coarse-to-fine learning paradigm** supporting hierarchical detail perception
- A **multi-region decomposition scheme** in which head and shoulders are predicted separately and integrated through cross-region combination

## Installation

1. Create a Python environment (Python 3.10+ recommended):

   ```bash
   conda create -n omgavatar python=3.12 -y
   conda activate omgavatar
   ```

2. Install the environment dependencies and the 3DGS renderer by following the instructions in [xg-chu/GAGAvatar](https://github.com/xg-chu/GAGAvatar).

3. Download the pre-trained checkpoint and place it under `ckpts/`:

   ```
   ckpts/omg_ckpt.pt
   ```

## Prepare Resources

Before running inference on a custom video, execute the tracking script to generate FLAME parameters and per-frame metadata:

```bash
sh scripts/track_video.sh
```

Edit the `-v` argument to point to your own video, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python core/libs/GAGAvatar_track/track_video.py \
    -v ./data/videos/obama.mp4
```

After tracking, the corresponding folder under `data/videos/<name>/` will contain `img_lmdb/`, `optim.pkl`, `smoothed.pkl`, etc., which can be used as a driver during inference.

## Quick Start

```bash
python inference.py \
    -i ./data/images/leijun.jpg \
    -d ./data/videos/obama \
    -sub 2 \
    -r ./ckpts/omg_ckpt.pt \
    --shoulder_enhance
```

- `-sub` can be set to `0`, `1`, or `2` to control the level of detail.
- Enabling `--shoulder_enhance` improves the shoulder region but slows down inference. The quantitative results and speeds reported in the paper are obtained **without** shoulder enhancement.

Results are written to `render_results/<MODEL_NAME>/`.

## Citation

If you find OMG-Avatar useful for your research or applications, please cite our work:

```bibtex
@misc{ren2026omgavataroneshotmultilodgaussian,
  title={OMG-Avatar: One-shot Multi-LOD Gaussian Head Avatar},
  author={Jianqiang Ren and Lin Liu and Steven Hoi},
  year={2026},
  eprint={2603.01506},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.01506},
}
```

## Disclaimer

This project is for academic research purposes only and may not be used for any commercial purposes. Users must comply with all applicable laws and regulations, including the *Interim Measures for the Administration of Anthropomorphic Interactive Services*, the *Provisions on the Administration of Deep Synthesis Internet Information Services*, and the *Personal Information Protection Law*. It is strictly prohibited to use this project for deepfakes, portrait generation without consent, virtual companion services, or any other unlawful or unethical purposes. We assume no legal liability for any downstream use of this project.
