# MultiPositionIG-LLIE

This repository provides the training and testing code for our multi-position illumination-guided low-light image enhancement method. The model builds on RetinexFormer and incorporates MS-IFM, D-IGAR, and IGS. Options are provided for LOL-v1, LOL-v2-real, LOL-v2-synthetic, SMID, SDSD-Indoor, and LOL-Blur.

The implementation is based on the [official RetinexFormer repository](https://github.com/caiyuanhao1998/Retinexformer). Its original MIT license and attribution are retained in [LICENSE.txt](LICENSE.txt). The model class is `MultiPositionIlluminationGuidance`. The backbone file keeps the RetinexFormer name so the released weights load without renaming.

## 1. Environment

The code has been tested with Python 3.9, PyTorch 2.5.1, and torchvision 0.20.1. Install the matching PyTorch / torchvision pair for your platform, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Training requires a CUDA-capable environment. Checkpoint loading and small-image inference can also run on CPU; add `--cpu` to a test command for CPU evaluation.

## 2. Prepare datasets

Download [LOL-v1](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing) and [LOL-v2](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing) from the dataset links provided by RetinexFormer. Its README also provides [SMID and SDSD-Indoor download and preparation instructions](https://github.com/caiyuanhao1998/Retinexformer#2-prepare-dataset). Obtain LOL-Blur from the original dataset provider. Dataset images are not redistributed here.

Place the paired data as follows, or change `dataroot_lq` and `dataroot_gt` in the relevant option file:

```text
data/
  LOLv1/
    Train/{input,target}/
    Test/{input,target}/
  LOLv2/
    Real_captured/{Train,Test}/{Low,Normal}/
    Synthetic/{Train,Test}/{Low,Normal}/
  SMID/
    test_list.txt
    SMID_LQ_np/<scene>/<frame>.npy
    SMID_Long_np/<scene>/<reference>.npy
  SDSD/
    indoor_static_np/
      input/<scene>/<frame>.npy
      GT/<scene>/<frame>.npy
  LOL_Blur/
    train/{low_blur,high_sharp_scaled}/<scene>/<frame>.png
    test/{low_blur,high_sharp_scaled}/<scene>/<frame>.png
```

Copy [meta_info/SMID_test_list.txt](meta_info/SMID_test_list.txt) to `data/SMID/test_list.txt`; the SMID loader uses these 49 audited test scenes to separate test and training scenes. The SDSD-Indoor option uses `pair11,pair21,pair1,pair19,pair4,pair9` as its evaluation scenes. For LOL-Blur, low-light and target images must have identical relative `<scene>/<frame>.png` paths; the loader checks this before evaluation.

## 3. Testing

Six checkpoints are available from [Google Drive](https://drive.google.com/drive/folders/1mpr5IR6m-t-1nhXdo1KB72eKEh4DG2Tj?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1Do676ZPTTMF8IJAdbMwDQw?pwd=wuet) (extraction code `wuet`).

| Dataset | File |
| --- | --- |
| LOL-v1 | `LOL v1.pth` |
| LOL-v2-real | `LOL v2-real.pth` |
| LOL-v2-synthetic | `LOL v2-syn.pth` |
| SMID | `SMID.pth` |
| SDSD-Indoor | `SDSD-indoor.pth` |
| LOL-Blur | `LOL-Blur.pth` |

Place the `.pth` files under `checkpoints/` at the repository root, keeping these filenames. Hashes are in [WEIGHTS_SHA256.txt](WEIGHTS_SHA256.txt). The Google Drive copies were downloaded and matched the verified local weights by SHA-256; the Baidu share was checked for filenames and sizes, but its file bytes were not independently hashed.

RetinexFormer uses one option per dataset for training and passes a checkpoint separately for testing. This repository follows the same convention. Run these commands from the repository root:

```bash
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv1.yml --weights "checkpoints/LOL v1.pth"
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv2Real.yml --weights "checkpoints/LOL v2-real.pth"
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv2Synthetic.yml --weights "checkpoints/LOL v2-syn.pth"
python -m basicsr.test --opt Options/paper/MultiPositionIG_SMID.yml --weights "checkpoints/SMID.pth"
python -m basicsr.test --opt Options/paper/MultiPositionIG_SDSDIndoor.yml --weights "checkpoints/SDSD-indoor.pth"
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLBlur.yml --weights "checkpoints/LOL-Blur.pth"
```

The test entry point selects only the option's `val` dataset, loads the supplied checkpoint, saves outputs under `results/`, and computes PSNR and SSIM against paired ground truth. It never evaluates the option's `train` dataset. LPIPS and LOE are not computed by these commands. Full benchmark scores were not re-evaluated as part of this code-release check.

## 4. Training

Each dataset has one option in `Options/paper/`:

```bash
python -m basicsr.train --opt Options/paper/MultiPositionIG_LOLv1.yml
python -m basicsr.train --opt Options/paper/MultiPositionIG_LOLv2Real.yml
python -m basicsr.train --opt Options/paper/MultiPositionIG_LOLv2Synthetic.yml
python -m basicsr.train --opt Options/paper/MultiPositionIG_SMID.yml
python -m basicsr.train --opt Options/paper/MultiPositionIG_SDSDIndoor.yml
python -m basicsr.train --opt Options/paper/MultiPositionIG_LOLBlur.yml
```

The LOL training options were reconstructed from the Supplement and available source templates; they are not verified copies of the original LOL run files. The SMID, SDSD-Indoor, and LOL-Blur options follow the archived run settings, with portable dataset paths and the public model class name. The SDSD-Indoor option covers **stage 1 only**. The released `SDSD-indoor.pth` is the later stage-2 checkpoint, so running this stage-1 option alone does not reproduce that weight.

SMID and SDSD-Indoor stage 1 have `batch_size_per_gpu: 4` in their archived run options. Their `mini_batch_sizes: [8]` setting does not increase the actual DataLoader batch in this implementation. Training periodically evaluates on the configured evaluation partition for PSNR-based checkpoint selection; those selected-checkpoint scores are not independently held-out estimates.

## Acknowledgment

This code is derived from [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer): Cai et al., *Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement*, ICCV 2023.
