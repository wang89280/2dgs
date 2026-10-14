# DenseSfM

Official implementation of **DenseSfM** (CVPR 2025).

This repository contains code for SfM build and SfM refinement only. 
The Gaussian Splatting-based track extension proposed in the paper is not included in this release — please refer to the paper for details.

---

## Installation

### 0. Clone the repository

```bash
git clone --recursive https://github.com/IceTea-CV/DenseSfM-Refine.git
cd DenseSfM-Refine
```

The `--recursive` flag is required to fetch the `third_party/RoIAlign.pytorch` submodule.

### 1. Environment

This codebase requires **CUDA 11.8**. Install PyTorch with the matching CUDA version first:

```bash
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 2. Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Pretrained Weight

Download the pretrained model and place it under the `weight/` directory:

- [weight/mv_refinement.ckpt](https://drive.google.com/file/d/16s18nqxEdAP6hyBsXSU-2zs8IjjX4p9o/view?usp=sharing)

```
weight/
└── mv_refinement.ckpt
```

### 4. RoIAlign (CUDA extension)

```bash
cd third_party/RoIAlign.pytorch
pip install .
```

### 5. Custom COLMAP Build (required for SfM refinement)

The SfM refinement step relies on a custom-patched COLMAP that exposes additional bundle adjustment interfaces.  
Please follow the COLMAP build instructions from [DetectorFreeSfM](https://github.com/zju3dv/DetectorFreeSfM/blob/main/INSTALL.md).

System-level dependencies needed before building.

---

## Usage

There are two main entry points depending on your use case.

---

### Mode 1: Full Pipeline — Matching + SfM + Refinement

`run_full.py` runs the complete pipeline: feature matching with a selected matcher, COLMAP-based SfM, and multiview refinement.

```bash
python run_full.py \
    --config-name base \
    "neuralsfm.NEUSFM_coarse_matcher=ALIKED+LightGlue" \
    "dataset_name=eth3d_test_dataset" \
    "scene_list=[pipes]"
```

See `run_full_eth3d_test.sh` for a full example.

This script uses [Hydra](https://hydra.cc/) for configuration. Key overrides:

- `neuralsfm.NEUSFM_coarse_matcher` — matcher to use (`ALIKED+LightGlue`, `RoMa`, `loftr_official`, ...)
- `dataset_name` — subdirectory under `SfM_dataset/`
- `scene_list` — list of scene names to process, e.g. `"scene_list=[pipes,courtyard]"`

Full config and all defaults are in `hydra_configs/base.yaml`.

#### Triangulation mode vs. Full SfM mode

`neuralsfm.triangulation_mode` controls whether camera poses are estimated or treated as fixed:

**`triangulation_mode: False` (Full SfM)**  
Runs the complete pipeline: coarse matching → COLMAP incremental SfM (pose estimation + triangulation) → multiview refinement of both poses and 3D points.  
Use this when starting from scratch with no prior camera poses.

```bash
python run_full.py \
    --config-name base \
    "neuralsfm.NEUSFM_coarse_matcher=ALIKED+LightGlue" \
    "neuralsfm.triangulation_mode=False" \
    "dataset_name=custom_dataset" \
    "scene_list=[my_scene]"
```

**`triangulation_mode: True` (Triangulation / Refinement only)**  
Camera poses are fixed (loaded from `<scene>/poses/`). Only 3D point positions are refined by the multiview refinement network. Camera intrinsics must be provided via `<scene>/intrins/` (set `use_prior_intrin: True`).  
Use this when you already have reliable poses and only want to densify or refine the point cloud.

```bash
python run_full.py \
    --config-name base \
    "neuralsfm.NEUSFM_coarse_matcher=ALIKED+LightGlue" \
    "neuralsfm.triangulation_mode=True" \
    "use_prior_intrin=True" \
    "dataset_name=custom_dataset" \
    "scene_list=[my_scene]"
```

---

### Mode 2: Refinement Only

`run_refinement.py` skips matching and SfM entirely — it takes an **existing COLMAP reconstruction** and runs only the multiview refinement step on top of it. Useful when you already have a coarse COLMAP model and want to apply DenseSfM refinement standalone.

```bash
python run_refinement.py \
    --img_folder <path/to/images> \
    --colmap_coarse_dir <path/to/colmap/model/0> \
    --refined_colmap_dir <path/to/output/refined> \
    --config hydra_configs/base.yaml
```

See `run_refinement_eth3d_test.sh` for a full example.

- `--img_folder` — directory containing the input images
- `--colmap_coarse_dir` — path to the coarse COLMAP model (the `0/` subfolder from mapper output)
- `--refined_colmap_dir` — output directory for the refined model
- `--config` — YAML config to read `neuralsfm` and `colmap_cfg` settings from (default: `hydra_configs/base.yaml`)
- `--chunk_size` — override `neuralsfm.NEUSFM_refinement_chunk_size` from config
- `--triangulation_mode` — if set, fixes camera poses and refines 3D points only (same semantics as Mode 1)

---

## Custom Dataset

An example scene is available for download — extract it under the `SfM_dataset/` directory:

- [SfM_dataset/custom_dataset](https://drive.google.com/file/d/1vM3GNqdDn_JOxhvMz09qv1Aw8pqX1Baw/view?usp=sharing)

After extraction the layout should look like:

```
SfM_dataset/
└── custom_dataset/
    └── scene_example/
        ├── images/
        ├── poses/
        └── intrins/
```

### Directory structure

```
SfM_dataset/
└── custom_dataset/
    └── scene_example/
        ├── images/          # Input images (e.g. DSC_0501.JPG, ...)
        ├── poses/           # (Optional) Per-image camera pose — 4×4 world-to-camera matrix, one .txt per image
        └── intrins/         # (Optional) Per-image intrinsic matrix — 3×3 K matrix, one .txt per image
```

To use your own data, follow the same structure with any dataset name and scene name.

`poses/` and `intrins/` are only required when `use_prior_intrin: True` or `triangulation_mode: True`.  
Each `.txt` filename must match its corresponding image filename (without extension).

**poses format** — 4×4 world-to-camera matrix (row-major):
```
r00 r01 r02 t0
r10 r11 r12 t1
r20 r21 r22 t2
0   0   0   1
```

**intrins format** — 3×3 camera intrinsic matrix:
```
fx  0  cx
0  fy  cy
0   0   1
```

### Running on custom data

**Full SfM from scratch** (no prior poses):

```bash
python run_full.py \
    --config-name base \
    "neuralsfm.NEUSFM_coarse_matcher=ALIKED+LightGlue" \
    "neuralsfm.triangulation_mode=False" \
    "dataset_name=custom_dataset" \
    "scene_list=[scene_example]"
```

**Refinement with prior poses** (triangulation mode):

```bash
python run_full.py \
    --config-name base \
    "neuralsfm.NEUSFM_coarse_matcher=ALIKED+LightGlue" \
    "neuralsfm.triangulation_mode=True" \
    "use_prior_intrin=True" \
    "dataset_name=custom_dataset" \
    "scene_list=[scene_example]"
```

---

## Acknowledgements

This work builds upon the following excellent projects:

- [RoMa](https://github.com/Parskatt/RoMa)
- [DetectorFreeSfM](https://github.com/zju3dv/DetectorFreeSfM)
- [COLMAP](https://github.com/colmap/colmap)
- [hloc](https://github.com/cvg/Hierarchical-Localization)

We thank the authors for their contributions. Portions of their code have been modified and incorporated into this repository.

---

## Citation

```bibtex
@InProceedings{Lee_2025_CVPR,
    author    = {Lee, JongMin and Yoo, Sungjoo},
    title     = {Dense-SfM: Structure from Motion with Dense Consistent Matching},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {6404-6414}
}
```
