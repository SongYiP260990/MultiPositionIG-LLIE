# MultiPositionIG-LLIE

Training and testing code for the multi-position illumination-guided low-light image enhancement model described in our Neurocomputing manuscript. The model retains the RetinexFormer backbone and adds MS-IFM, D-IGAR, and IGS. Its paper-facing class is `MultiPositionIlluminationGuidance`; `RetinexFormer_arch.py` keeps the upstream name because the verified checkpoint parameter keys and implementation depend on that module.

This is a clean release of the code needed for the three LOL datasets below. It is based on the [RetinexFormer repository](https://github.com/caiyuanhao1998/Retinexformer); the original MIT copyright notice is retained in `LICENSE.txt`. The repository does not contain dataset images or model weights.

## Checkpoints

The author plans to provide the following three weights through Baidu Netdisk and Google Drive. **The download links have not been added yet.** The test options intentionally contain `REPLACE_WITH_...` paths; replace each path with the downloaded file's location before testing.

| Dataset | Checkpoint filename | Baidu Netdisk | Google Drive |
| --- | --- | --- | --- |
| LOL-v1 | `best_psnr_25.17_1000.pth` | Pending | Pending |
| LOL-v2-real | `best_psnr_23.67_22600.pth` | Pending | Pending |
| LOL-v2-synthetic | `best_psnr_26.38_205500.pth` | Pending | Pending |

These three files were checked for strict parameter loading with the released model class on CPU. That check establishes model compatibility; it does not independently reproduce the paper's test-set scores. Weights for other datasets are outside this initial release.

Expected SHA-256 hashes for the local files verified during release preparation (compare them after downloading):

```text
LOL-v1          DCEFB7062190C2CA987C337FE42ECBFA7E1274C11996B58D883D8BFD641EED41
LOL-v2-real     CB3092F2BC559B6A1DFD2B12142BB3FF74AC60E1667DF46941E6005926ACB249
LOL-v2-synthetic 999C679BEBC6DACC073BFBC26B41A638CB74BD64F48F4017CB739E1C814D1C88
```

## Environment

Use Python 3.9 or a compatible version. Install a matching PyTorch and torchvision build for your machine, then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

The code was import-checked locally with Python 3.9, PyTorch 2.5.1, and torchvision 0.20.1. Training requires a CUDA-capable setup; checkpoint loading and a small inference smoke test can run on CPU. `tensorboard` is used by the training logger.

## Data layout

Obtain LOL-v1 and LOL-v2 from their dataset distributors; the image data are not redistributed here. Set the `dataroot_lq` and `dataroot_gt` fields in the selected YAML if your layout differs. The supplied options expect:

```text
data/
  LOLv1/{Train,Test}/{input,target}/
  LOLv2/Real_captured/{Train,Test}/{Low,Normal}/
  LOLv2/Synthetic/{Train,Test}/{Low,Normal}/
```

Each low-light image must have a corresponding normal-light file under the paired directory. Check the dataset's original terms before use. The upstream [RetinexFormer instructions](https://github.com/caiyuanhao1998/Retinexformer) provide dataset acquisition and preparation context.

## Test

Run commands from the repository root. Edit `path.pretrain_network_g` in the matching `*_test.yml` to point to your downloaded `.pth` file. Test options have only a `test` dataset entry; they are separate from training options.

```bash
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv1_test.yml
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv2Real_test.yml
python -m basicsr.test --opt Options/paper/MultiPositionIG_LOLv2Synthetic_test.yml
```

The test entry point loads the checkpoint strictly and computes PSNR and SSIM against paired ground truth. Enhanced images and logs go under `results/`. Other paper metrics, including LPIPS and the separately specified LOE protocol, are not computed by these commands.

## Train

The corresponding training options are in `Options/paper/`. For example:

```bash
python -m basicsr.train --opt Options/paper/MultiPositionIG_LOLv2Real_train.yml
```

The options encode the paper's dataset, patch size, batch size, iteration budget, learning rate, loss weights, and MixUp settings. They were assembled from the supplementary tables and available source templates; **they are release configurations, not verified copies of each checkpoint's original run YAML**. In particular, the LOL-v1 option follows the reported 320-pixel patch, batch size 4, 300k-iteration, `2e-4` learning rate, `(1.0, 0.5, 0.05)` loss-weight, MixUp configuration and is not claimed as the exact YAML that generated `best_psnr_25.17_1000.pth`.

The training entry point evaluates periodically on the configured paired evaluation partition for checkpoint selection; there is no separate official validation split in these options. Do not treat those measurements as an untouched test set for a newly trained run.

## Code map

- `basicsr/models/archs/RetinexFormer_arch.py`: backbone and proposed modules.
- `basicsr/models/archs/MultiPositionIlluminationGuidance_arch.py`: paper-facing model name with checkpoint-compatible keys.
- `basicsr/models/losses/losses.py`: training objectives.
- `basicsr/train.py` and `basicsr/test.py`: training and paired testing entry points.
- `Options/paper/`: three dataset-specific training and test configurations.
