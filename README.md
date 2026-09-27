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

Place the `.pth` files under `checkpoints/` at the repository root, keeping these filenames.

RetinexFormer uses one option per dataset for training and passes a checkpoint separately for testing. This repository follows the same convention. Run these commands from the repository root:

```bash
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv1.yml --weights "checkpoints/LOL v1.pth"
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv2Real.yml --weights "checkpoints/LOL v2-real.pth"
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv2Synthetic.yml --weights "checkpoints/LOL v2-syn.pth"
python -m basicsr.test --opt Options/paper/MultiPositionIG_SMID.yml --weights "checkpoints/SMID.pth"
python -m basicsr.test --opt Options/paper/MultiPositionIG_SDSDIndoor.yml --weights "checkpoints/SDSD-indoor.pth"
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLBlur.yml --weights "checkpoints/LOL-Blur.pth"
```

The test commands load the supplied checkpoints, save enhanced images under `results/`, and compute PSNR and SSIM against the paired ground truth. LPIPS and LOE are not computed by these commands.

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

Checkpoint selection uses PSNR. The SDSD-Indoor result uses the checkpoint selected at iteration 36k of the 50k training run.

## Acknowledgment

This code is derived from [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer): Cai et al., *Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement*, ICCV 2023.
