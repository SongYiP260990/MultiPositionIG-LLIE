# MultiPositionIG-LLIE

This repository provides the training and testing code for our multi-position illumination-guided low-light image enhancement method. The model builds on RetinexFormer and incorporates MS-IFM, D-IGAR, and IGS. The released options cover LOL-v1, LOL-v2-real, and LOL-v2-synthetic.

The implementation is based on the [official RetinexFormer repository](https://github.com/caiyuanhao1998/Retinexformer). Its original MIT license and attribution are retained in [LICENSE.txt](LICENSE.txt). The paper-facing model class is `MultiPositionIlluminationGuidance`; the backbone file retains its RetinexFormer name to preserve compatibility with the trained weights.

## 1. Environment

The tested CPU environment uses Python 3.9, PyTorch 2.5.1, and torchvision 0.20.1. Install the matching PyTorch pair for your platform, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The code was checked locally with Python 3.9, PyTorch 2.5.1, and torchvision 0.20.1. Training requires a CUDA-capable environment; checkpoint loading and small-image inference can also run on CPU.

## 2. Prepare datasets

Download [LOL-v1](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing) and [LOL-v2](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing) from the dataset links provided by RetinexFormer. Their README also provides [Baidu Disk links and preparation details](https://github.com/caiyuanhao1998/Retinexformer#2-prepare-dataset). Dataset images are not redistributed here.

Organize the paired images as follows, or change `dataroot_lq` and `dataroot_gt` in the relevant option file:

```text
data/
  LOLv1/
    Train/{input,target}/
    Test/{input,target}/
  LOLv2/
    Real_captured/
      Train/{Low,Normal}/
      Test/{Low,Normal}/
    Synthetic/
      Train/{Low,Normal}/
      Test/{Low,Normal}/
```

The low-light and normal-light directories must contain matching image filenames.

## 3. Testing

Six checkpoints are available from [Google Drive](https://drive.google.com/drive/folders/1mpr5IR6m-t-1nhXdo1KB72eKEh4DG2Tj?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1Do676ZPTTMF8IJAdbMwDQw?pwd=wuet) (extraction code `wuet`). The Google Drive copies were downloaded and their SHA-256 hashes matched the verified local weights. The Baidu share was opened with the extraction code; its six filenames and sizes matched, but its file bytes were not independently hashed. The weights are not stored in Git.

Place the downloaded `.pth` files directly in `checkpoints/` at the repository root, retaining the archive filenames. Verify them against [WEIGHTS_SHA256.txt](WEIGHTS_SHA256.txt) before use.

| Dataset | Archive filename | Release scope |
| --- | --- | --- |
| LOL-v1 | `LOL v1.pth` | Strict loading and small-image inference verified; full benchmark score not re-evaluated for this release |
| LOL-v2-real | `LOL v2-real.pth` | Strict loading and small-image inference verified; this exact weight was used for the revision gate analysis; full benchmark score not re-evaluated for this release |
| LOL-v2-synthetic | `LOL v2-syn.pth` | Strict loading and small-image inference verified; full benchmark score not re-evaluated for this release |
| SMID | `SMID.pth` | Paper checkpoint identified by training and evaluation records; dataset-specific test option not included here |
| SDSD-Indoor | `SDSD-indoor.pth` | Two-stage Table 4 checkpoint identified by training and evaluation records; the separate component ablation uses different runs; dataset-specific test option not included here |
| LOL-Blur | `LOL-Blur.pth` | Paper checkpoint identified by training and evaluation records; dataset-specific test option not included here |

All six weights have the same 40-channel, one-stage, `[1,2,2]`-block model graph. The commands below cover the three LOL datasets for which this repository includes dataset-specific options; the other released weights are provided for inspection and future dataset-specific evaluation, not as a claim that this repository contains a ready-to-run benchmark pipeline for those datasets.

Run these commands from the repository root:

```bash
# LOL-v1
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv1_test.yml

# LOL-v2-real
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv2Real_test.yml

# LOL-v2-synthetic
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv2Synthetic_test.yml
```

The test commands load the corresponding shared checkpoint strictly, save enhanced images under `results/`, and compute PSNR and SSIM against paired ground truth. LPIPS and LOE are not computed by these commands. The three LOL weights passed a CPU loading and single-image inference check with this code; full-dataset paper scores were not re-evaluated as part of this release check.

## 4. Training

The corresponding training options are in `Options/paper/`:

```bash
# LOL-v1
python -m basicsr.train --opt Options/paper/MultiPositionIG_LOLv1_train.yml

# LOL-v2-real
python -m basicsr.train --opt Options/paper/MultiPositionIG_LOLv2Real_train.yml

# LOL-v2-synthetic
python -m basicsr.train --opt Options/paper/MultiPositionIG_LOLv2Synthetic_train.yml
```

These release options were assembled from the paper's supplementary training settings and available source templates. They specify the reported patch sizes, batch sizes, iteration budgets, learning rates, objectives, and MixUp settings; they are **not verified copies of the original run YAML files**. The LOL-v1 option in particular follows the reported supplementary settings rather than a recovered original run configuration.

The training entry point evaluates periodically on the configured paired evaluation partition for checkpoint selection. These options do not define a separate official validation split, so a newly trained run's selection measurements should not be described as untouched test evidence.

## Acknowledgment

This code is derived from [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer): Cai et al., *Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement*, ICCV 2023.

