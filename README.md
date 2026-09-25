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

Checkpoint download links are pending. The testing commands below require the corresponding weight file in `checkpoints/`; they cannot run until the weights are obtained. The test options already name the three intended files:

| Dataset | Expected file | Architecture | Paper-result status |
| --- | --- | --- | --- |
| LOL-v1 | `best_psnr_25.17_1000.pth` | 40 channels, `[1,2,2]` blocks | Selected checkpoint; benchmark score has not been re-evaluated in this release check |
| LOL-v2-real | `best_psnr_23.67_22600.pth` | 40 channels, `[1,2,2]` blocks | Selected checkpoint used in the revision gate analysis; benchmark score has not been re-evaluated in this release check |
| LOL-v2-synthetic | `best_psnr_26.38_205500.pth` | 40 channels, `[1,2,2]` blocks | Selected checkpoint; benchmark score has not been re-evaluated in this release check |

These files are not stored in Git. Download locations will be added after the archives are uploaded and access is verified. SMID and LOL-Blur weights are withheld pending provenance checks. The SDSD-Indoor organized weight uses a different checkpoint-compatible architecture from the current release training option, so it is also withheld.

Run these commands from the repository root:

```bash
# LOL-v1
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv1_test.yml

# LOL-v2-real
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv2Real_test.yml

# LOL-v2-synthetic
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv2Synthetic_test.yml
```

The test commands load the selected checkpoint strictly, save enhanced images under `results/`, and compute PSNR and SSIM against paired ground truth. LPIPS and LOE are not computed by these commands. The three intended weights passed a CPU loading and single-image inference check with this code; full-dataset paper scores were not re-evaluated as part of this release check.

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

