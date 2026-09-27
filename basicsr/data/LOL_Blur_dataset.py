from torch.utils import data as data
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
import torchvision.transforms.functional as TF
from typing import Sequence

from PIL import Image

import random
import torch
import os


class RandomRotation(torch.nn.Module):
    def __init__(self, angles: Sequence[int]):
        super().__init__()
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class Dataset_LOLBlurImage(data.Dataset):
    def __init__(self, opt):
        super(Dataset_LOLBlurImage, self).__init__()
        self.opt = opt
        gt_folder, lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.lq_paths, self.gt_paths = self._get_imgs_path(lq_folder, gt_folder)
        self.to_tensor = ToTensor()

        if self.opt.get('phase') == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)
            self.random_crop = RandomCrop([opt['gt_size'], opt['gt_size']])
            if self.geometric_augs:
                self.transform = Compose([
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomRotation([0, 90, 180, 270])
                ])

    def _flatten_list_comprehension(self, matrix):
        return [item for row in matrix for item in row]

    def _get_imgs_path(self, lq_path, gt_path):
        """Get paired image paths (scene_id/frame.png), sorted for stable pairing."""
        def list_scene_pngs(root):
            scenes = sorted(
                [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            )
            paths = []
            for scene in scenes:
                scene_dir = os.path.join(root, scene)
                pngs = sorted([p for p in os.listdir(scene_dir) if p.lower().endswith('.png')])
                for name in pngs:
                    paths.append(os.path.join(scene_dir, name))
            return paths

        lq_paths = list_scene_pngs(lq_path)
        gt_paths = list_scene_pngs(gt_path)
        if len(lq_paths) != len(gt_paths):
            raise ValueError(
                f'lq/gt count mismatch: lq={len(lq_paths)} gt={len(gt_paths)} '
                f'({lq_path} vs {gt_path})'
            )
        lq_keys = [os.path.relpath(path, lq_path) for path in lq_paths]
        gt_keys = [os.path.relpath(path, gt_path) for path in gt_paths]
        if lq_keys != gt_keys:
            mismatch = next((i for i, (lq, gt) in enumerate(zip(lq_keys, gt_keys))
                             if lq != gt), None)
            raise ValueError(
                f'lq/gt relative paths differ at index {mismatch}: '
                f'{lq_keys[mismatch]} vs {gt_keys[mismatch]}')
        return lq_paths, gt_paths

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        img_gt = self.to_tensor(Image.open(gt_path).convert('RGB'))

        lq_path = self.lq_paths[index]
        img_lq = self.to_tensor(Image.open(lq_path).convert('RGB'))

        if self.opt.get('phase') == 'train':
            high_and_low = torch.stack((img_gt, img_lq))
            high_and_low = self.random_crop(high_and_low)
            if self.geometric_augs:
                high_and_low = self.transform(high_and_low)
            img_gt, img_lq = high_and_low

        return {
            'lq': img_lq.float(),
            'gt': img_gt.float(),
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.gt_paths)
