# OMG-Avatar: One-shot Multi-LOD Gaussian Head Avatar

[![CVPR 2026](https://img.shields.io/badge/project-page-blue)](https://human3daigc.github.io/OMGAvatar_project_page/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.01506-b31b1b.svg)](https://arxiv.org/abs/2603.01506)

This repository contains the codes for **OMG-Avatar**, accepted to **CVPR 2026**.

## Overview

We propose a unified model that accommodates diverse hardware capabilities and inference speed requirements through LOD head avatar modeling. Our method employs:

- A **transformer-based architecture** for global feature extraction
- **Projection-based sampling** for local feature acquisition
- **Depth-buffer-guided fusion** ensuring occlusion plausibility
- A **coarse-to-fine learning paradigm** supporting hierarchical detail perception
- A **multi-region decomposition scheme** where head and shoulders are predicted separately and integrated through cross-region combination



## Code

**The code is currently being organized and will be released soon. Stay tuned!**

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

