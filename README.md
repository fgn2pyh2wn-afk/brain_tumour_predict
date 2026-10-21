[README.md](https://github.com/user-attachments/files/33765805/README.md)

# From a Single Scan to a Trajectory: Two-Stage Latent Diffusion for Controllable Local Progression Synthesis of Brain Metastases

[![Paper](https://img.shields.io/badge/论文-中文手稿-blue)](#) [![I# Brain Metastasis Progression Synthesis with a Lightweight Latent Diffusion Transformer
**From a single T1-Gd scan to a bounded, reproducible 3D progression sequence.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![Paper](https://img.shields.io/badge/CV4DC%202026-Non--Proceedings%20Track-purple)](https://cv4dc.github.io/2026/)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)](#intended-use-and-limitations)

This repository contains the code, checkpoints, notebooks and reproducibility scripts for a
**two-stage latent diffusion model** that synthesises a temporal progression sequence of a brain
metastasis from a **single T1-Gd MRI volume**. The pipeline is deliberately small — fewer than
10M trainable parameters in total — and is designed for **resource-constrained research settings**:
it trains in seconds per epoch on one commodity GPU, runs on Apple Silicon during development, and
writes standard NIfTI volumes that free viewers can open.
 **Research use only.** This is not a medical device and must not be used for diagnosis, treatment
> planning, or any clinical decision. See [Intended Use and Limitations](#intended-use-and-limitations).
<img width="2325" height="789" alt="6ae34af8df09c0c72be25c82fb55fb34" src="https://github.com/user-attachments/assets/9f9eec31-518e-4424-ac3d-555368b24850" />>---

## Table of Contents

- [Highlights](#highlights)
- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Checkpoints](#checkpoints)
- [Results](#results)
- [Ablations](#ablations)
- [Evaluation Protocol](#evaluation-protocol)
- [Training Details](#training-details)
- [Reproducibility Notes](#reproducibility-notes)
- [Intended Use and Limitations](#intended-use-and-limitations)
- [Citation](#citation)
- [License and Data Ethics](#license-and-data-ethics)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- **Two-stage latent diffusion.** A residual 3D VAE compresses a `64³` T1-Gd volume into a
  `4×16³` latent; a Diffusion Transformer (DiT) with adaLN-Zero conditioning and a cosine noise
  schedule denoises the latent; two-stage DDIM sampling produces the progression sequence.
- **Explicit temporal control.** A dark-lesion growth prior converts deterministic sampling into a
  **monotone** progression: the hypointense lesion deepens and dilates as the generation step
  increases, instead of relying on an implicit interpolation.
- **Small and fast.** 5.52M parameters (VAE) + 4.23M parameters (DiT); **7 s/epoch** on one
  NVIDIA RTX PRO 6000 Blackwell GPU; the full training + evaluation + ablation suite runs in
  ~28 minutes.
- **Verified numbers.** Every metric in this README was produced by the released code. The VAE
  reaches **34.29 dB PSNR / 0.9829 SSIM** on the held-out split; the DiT reaches **0.1981** test
  latent-noise MSE after phase 1 and **0.2254** after fine-tuning.
- **Honest ablations.** Four controlled ablations isolate the KL term, EMA, the final LayerNorm
  initialisation, and the growth prior — including a **measured negative result**: a naive
  adaLN-Zero port silently cuts the gradient path and pins the DiT at "predict-nothing"
  (MSE ≈ 1.0).

## Method Overview

```
Input: single T1-Gd MRI (64³)
        │
        ▼
Stage 1 · 3D VAE ──► latent (4×16³, standardised)
        │                  │
        │                  └── pooled latent ──► condition
        ▼
Stage 2 · DiT (dim 192, 6 blocks, adaLN-Zero, cosine T=1000, EMA)
        │
        ▼
Two-stage DDIM sampling (50 steps each)
   • stage 1: AdaNI sets the starting noise level from the generation step
   • stage 2: re-noise to 0.12T and refine texture
        │
        ▼
Post-processing: brain-mask clip ±0.18 + dark-lesion growth prior
        │
        ▼
Output: 9-step progression sequence (NIfTI + PNG)
```

For a visual overview, see `核验材料/ISBI单页/fig_framework.png` and the CV4DC 2026 paper in
`CV4DC2026投稿/`.

## Repository Structure

```text
brain_tumour_predict/
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_all.py              # main training + evaluation + all four ablations
│   └── gen_more_demos.py       # multi-case qualitative demo sequences
├── 脑肿瘤预测_优化实验.ipynb        # baseline vs. optimised experiment (executed, with figures)
├── 脑肿瘤预测_时序演化.ipynb        # full temporal progression demo (executed, with figures)
├── 脑肿瘤预测_DiT算法.ipynb        # main DiT experiment (executed)
├── 脑转移瘤局部进展演化的两阶段潜空间扩散模型DiT.docx   # manuscript
├── CV4DC2026投稿/               # ACCV/CV4DC 2026 submission sources (LaTeX, LNCS)
│   ├── main.tex
│   ├── main.pdf
│   └── fig_*.png
├── 核验材料/                    # verification materials, poster figures, ISBI one-pager
│   └── ISBI单页/
├── 复算脚本/                    # independent audit and recomputation scripts
├── 备份/                        # versioned backups (not required to run)
├── data/                        # auxiliary checkpoints (.pt)
├── 实验图_01_重建与扩散对比.png
├── 实验图_02_时序演化_原始.png
├── 实验图_03_时序演化_分割增强.png
├── 实验图_04_病灶增长曲线.png
└── brain_tumor_ldm.pt           # checkpoint (see Checkpoints; regenerate with run_all.py)
```

> The **dataset is not distributed** with this repository. See [Data Preparation](#data-preparation).

## Installation

### Requirements

- Python 3.10+ (3.12 tested)
- PyTorch 2.x (CUDA 12.x recommended; CPU and Apple Silicon MPS also work)
- `numpy`, `matplotlib`, `Pillow`, `nibabel`
- Optional: `scikit-image` (only for cross-checking SSIM)

### Setup

```bash
git clone https://github.com/fgn2pyh2wn-afk/brain_tumour_predict.git
cd brain_tumour_predict

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Install the PyTorch build that matches your CUDA/driver, e.g.
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify the environment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Data Preparation

We use the publisher-provided **105/51 train/test split** of
[BrainMetShare](https://aimi.stanford.edu/datasets) (Stanford AIMI). The dataset is
**not included** here and must be obtained under its own licence (non-commercial research use).

Expected layout (one directory per case, one sub-directory per modality):

```text
BRAINMETASTASIS/
├── train/
│   ├── Mets_001/
│   │   ├── t1_gd/        # *.png slices, e.g. t1_gd_z001.png ...
│   │   ├── t1_pre/
│   │   ├── flair/
│   │   ├── bravo/
│   │   └── seg/          # expert segmentation (training split only)
│   └── ...
└── test/
    ├── Mets_051/
    │   └── t1_gd/
    └── ...
```

The pipeline reads `t1_gd` slices, resizes them to `64×64` in-plane, and trilinearly resamples each
case to a `64³` volume. All slices are cached in memory once at start-up.

**macOS users:** remove AppleDouble metadata files before training; otherwise `*.png` globs pick up
`._*` twins and image loading fails:

```bash
find /path/to/BRAINMETASTASIS -name '._*' -delete
```

## Quick Start

### 1. Train, evaluate, and run all ablations

```bash
export BT_DATA=/path/to/BRAINMETASTASIS     # dataset root (contains train/ and test/)
export BT_HOME=/path/to/outputs             # checkpoints, results.json, demo images

python scripts/run_all.py
```

This single command:

1. caches the dataset (105 train / 51 test cases);
2. trains the 3D VAE for **30 epochs** and evaluates reconstruction (PSNR / SSIM / MSE / MAE);
3. trains the DiT for **40 epochs** (phase 1) and measures test latent-noise MSE;
4. fine-tunes for **40 more epochs** (phase 2, lr `1e-4`) and measures the final test MSE;
5. runs four ablations: VAE without KL, DiT without EMA, DiT with a zero-initialised final
   LayerNorm (the negative result), and the growth prior on/off;
6. generates the 9-step Mets_020 sequence with and without the growth prior;
7. writes `$BT_HOME/results.json` and `$BT_HOME/brain_tumor_ldm.pt`.

Expected runtime: ~28 minutes on a single RTX PRO 6000 Blackwell; ~7 s per epoch.

### 2. Generate additional qualitative demos

```bash
python scripts/gen_more_demos.py
```

This loads `$BT_HOME/brain_tumor_ldm.pt` (run `scripts/run_all.py` first if the bundled checkpoint
reports incompatible keys), samples cases `Mets_006`, `Mets_021`, `Mets_033`, and writes 9-step
axial strips to `$BT_HOME/demo_out/`.

### 3. Use the notebooks

The notebooks are self-contained and auto-detect the environment (`cuda` → `mps` → `cpu`) and the
dataset location (`/root/autodl-tmp` on cloud instances, otherwise `/Users/chris/Desktop`):

- `脑肿瘤预测_优化实验.ipynb` — baseline vs. optimised metrics, training curves, reconstructions;
- `脑肿瘤预测_时序演化.ipynb` — full temporal progression, growth curve, multi-view figures;
- `脑肿瘤预测_DiT算法.ipynb` — the main experiment with executed outputs.

To run on a remote server, open the notebook in JupyterLab (or VS Code Remote‑SSH) and select the
Python kernel. The notebooks write figures inline and NIfTI sequences to `$BT_HOME/tumor_sim_out/`.

## Checkpoints

`brain_tumor_ldm.pt` is a dictionary with:

| Key | Content |
| --- | --- |
| `vae` | `state_dict` of the residual 3D VAE (including latent-normalisation buffers) |
| `unet` | `state_dict` of the DiT denoiser |
| `betas` | cosine noise schedule (length 1000) |
| `T` | number of diffusion steps (1000) |

Load it with:

```python
import torch
ckpt = torch.load("brain_tumor_ldm.pt", map_location="cpu", weights_only=True)
```

> **Checkpoint compatibility.** The file shipped in this repository was produced by an earlier
> iteration of the pipeline and may not load into the current `VAE3DOpt` / `DiT3DOpt` definitions.
> If `load_state_dict` reports missing or unexpected keys, run `python scripts/run_all.py` once;
> it writes a matching checkpoint to `$BT_HOME/brain_tumor_ldm.pt`. Copy that file here (or point
> `BT_HOME` at your output directory) before running the demo script.

## Results

All numbers below come from a fresh end-to-end run of the released code (fixed seed 42,
12 of 51 test cases, `64³` volumes, one NVIDIA RTX PRO 6000 Blackwell GPU, PyTorch 2.8.0+cu128).

### Reconstruction and diffusion accuracy

| Stage | PSNR (dB) | SSIM | MSE | Params |
| --- | ---: | ---: | ---: | ---: |
| 3D VAE, 30 epochs, KL annealed | **34.29** | **0.9829** | 3.72 × 10⁻⁴ | 5.52M |
| DiT, phase 1 (40 epochs) | — | — | **0.1981** (test latent MSE) | 4.23M |
| DiT, phase 2 fine-tune (40 epochs) | — | — | 0.2254 (test latent MSE) | 4.23M |

> The fine-tune lowers training MSE (0.2174 → 0.2014) but slightly raises test MSE on this small
> dataset — a mild overfitting signal that we report rather than hide.

### Temporal progression (case Mets_020)

With the dark-lesion growth prior, newly-darkened voxels grow monotonically across nine steps:

```text
0 → 1022 → 1098 → 1491 → 2028 → 2390 → 3452 → 5563 → 5776
```

Without the prior, the same checkpoint produces a non-monotone, smaller change:

```text
0 → 873 → 1034 → 1365 → 1227 → 2031 → 2747 → 3305 → 3596
```

## Ablations

Same seed, data, and evaluation protocol; full script in `scripts/run_all.py`.

| Component | Variant | Metric | Value |
| --- | --- | --- | --- |
| VAE KL term | with KL (main) | PSNR / SSIM | 34.29 dB / 0.9829 |
| | **without KL** | PSNR / SSIM | **35.87 dB / 0.9847** |
| DiT EMA | with EMA (main) | test MSE | **0.1981** |
| | without EMA | test MSE | 0.2232 |
| DiT final norm | parameter-free (main) | test MSE | **0.1981** |
| | zero-initialised affine | test MSE | 0.9977 ("predict-nothing") |
| Growth prior | with prior (main) | dark voxels at step 8 | **5776** |
| | without prior | dark voxels at step 8 | 3596 (non-monotone) |

**Takeaways**

1. The KL term is slightly harmful for this small VAE (+1.57 dB without it); consider disabling it
   when reconstruction quality is the priority.
2. EMA is worth ~0.025 test MSE.
3. Zero-initialising the final LayerNorm together with the output projection creates two zero
   matrices in series and **cuts the entire backward path** — the DiT never leaves zero output
   (MSE ≈ 1.0 for 40 epochs). Use a parameter-free final norm.
4. The monotone progression comes from the **growth prior**, not from the sampler alone. This is an
   important honesty caveat: do not interpret the sequence as sampled tumour growth.

## Evaluation Protocol

- **Split:** publisher-provided 105/51 train/test split of BrainMetShare.
- **Evaluation sample:** 12 of 51 test cases, chosen once with a fixed seed (42).
- **Reconstruction metrics:** MSE, MAE, PSNR (`-10·log10(MSE)`), and 3D SSIM (Gaussian window 11,
  σ = 1.5) on `64³` volumes.
- **Diffusion metric:** MSE of predicted versus injected latent noise, averaged over the evaluation
  sample (one random timestep per case).
- **Temporal metric:** number of voxels that were not dark in step 0 (`≥ 0.22`) but became dark
  (`< 0.22`) at each step, inside the brain mask.

## Training Details

| Component | Setting |
| --- | --- |
| VAE | residual blocks, GroupNorm + SiLU, channels 32→64→128, latent 4×16³, KL annealed to 1e-3, EMA latent normalisation, 30 epochs, AdamW 3e-4, batch 4 |
| DiT | dim 192, 6 blocks, 6 heads, patch 2 (512 tokens), sinusoidal time embedding, adaLN-Zero, cosine schedule T=1000, EMA 0.999 |
| Phase 1 | 40 epochs, AdamW 2e-4, gradient clipping 1.0 |
| Phase 2 | 40 epochs fine-tune, AdamW 1e-4, EMA 0.999 |
| Sampling | two-stage DDIM, 50 steps each; AdaNI start level `T(0.06 + 0.70·k/K)`; refine at `0.12T` |
| Post-processing | brain-mask clip ±0.18; dark-lesion prior (30th-percentile hypointensity, Gaussian localisation, darken + dilate by `p(k)`) |

## Reproducibility Notes

- **Seeds:** Python, NumPy, and PyTorch seeds are fixed to 42 in `scripts/run_all.py`.
- **Caching:** all 109,350 PNG slices are loaded once (~10 s on a fast CPU); epochs then avoid disk I/O.
- **AppleDouble files:** delete `._*` twins before training (see [Data Preparation](#data-preparation)).
- **Device:** `cuda` → `mps` → `cpu` is selected automatically; MPS is supported for development,
  CUDA for full runs. Some 3D pooling ops may fall back to CPU on older MPS versions
  (`PYTORCH_ENABLE_MPS_FALLBACK=1`).
- **Non-determinism:** GPU reductions may introduce small run-to-run differences in the third decimal
  of PSNR/MSE; all reported numbers are from actual runs, not synthetic values.

## Intended Use and Limitations

**Intended use**

- Research on generative models for scarce medical imaging data;
- Education and method development in resource-constrained settings;
- Data augmentation experiments (with appropriate safeguards and expert oversight).

**Out of scope**

- Clinical diagnosis, treatment planning, or patient communication;
- Any claim that the synthesised sequence corresponds to real tumour growth;
- Use on data outside the licensed BrainMetShare cohort without validation.

**Limitations**

- No longitudinal ground truth: the step index is a generation condition, not clinical time.
- The lesion mask is a hypointensity heuristic, not an expert segmentation.
- Volumes are resampled to `64³`; voxel spacing is not modelled.
- Metrics are computed on 12 of 51 test cases with a fixed seed.
- The fine-tuned checkpoint slightly overfits the small training set (test MSE 0.2254 vs. 0.1981
  after phase 1).

## Citation

If you use this code or the generated sequences, please cite:

```bibtex
@inproceedings{he2026brainmet,
  title     = {A Lightweight Two-Stage Latent Diffusion Transformer for
               Brain-Metastasis Progression Synthesis under Resource Constraints},
  author    = {He, Jinyu and Lin, Zhijie and Wang, Kai},
  booktitle = {Workshop on Computer Vision for Developing Countries (CV4DC),
               ACCV 2026},
  year      = {2026},
  note      = {Non-Proceedings Track}
}
```

## License and Data Ethics

- **Code:** released for non-commercial research use. Add a `LICENSE` file (e.g., MIT or
  Apache-2.0) if you intend to permit broader reuse.
- **Data:** BrainMetShare is **not** redistributed here. It is governed by the publisher's
  non-commercial research licence; obtain it from the official source and follow its terms.
- **Ethics:** retrospective, de-identified secondary analysis of a public research dataset. No new
  patient data were collected. Synthetic outputs must not be presented as real follow-up imaging.

## Acknowledgements

- [BrainMetShare](https://aimi.stanford.edu/datasets) (Stanford AIMI) for the dataset.
- The open-source PyTorch, NumPy, Matplotlib, Pillow and NiBabel communities.
- Computational resources: NVIDIA RTX PRO 6000 Blackwell (AutoDL), with development on Apple
  Silicon (MPS).

---

**Contact:** open an issue on GitHub for questions about the code or reproduction.
SBI](https://img.shields.io/badge/ISBI-单页摘要-green)](#) [![Repro](https://img.shields.io/badge/复算核对-PASS%20220%20%2F%20FAIL%200-brightgreen)](#9-复算与核验)


贺晋羽 · GitHub: [@fgn2pyh2wn-afk](https://github.com/fgn2pyh2wn-afk)
<img width="2500" height="886" alt="f24643c7921f1911e846a8cf656316fa" src="https://github.com/user-attachments/assets/f6705b76-e712-4096-9b48-85c0b9bce6c5" />

---



