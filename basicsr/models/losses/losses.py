import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from basicsr.models.losses.loss_util import weighted_loss
import math
_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss   #把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss   #把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
# =========================================================
#  Final "PSNR-Friendly" Color Loss
#  Optimized for LOL-v2-real: Soft Mask + Scale Norm
# =========================================================

class ColorLoss(nn.Module):
    """
    Color Consistency Loss with Soft Luminance Gating.
    """
    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-6, dark_threshold=0.03):
        super(ColorLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps
        self.dark_threshold = dark_threshold

    def forward(self, pred, target):
        # pred, target: [B, C, H, W]

        # 1. Cosine Similarity
        pred_norm = torch.norm(pred, dim=1, keepdim=True) + self.eps
        target_norm = torch.norm(target, dim=1, keepdim=True) + self.eps

        pred_vec = pred / pred_norm
        target_vec = target / target_norm

        cosine_sim = torch.sum(pred_vec * target_vec, dim=1, keepdim=True)

        # 2. Soft Mask (Critical for Stability)
        # Smooth transition from 0 (dark) to 1 (bright)
        target_lum = torch.mean(target, dim=1, keepdim=True)
        mask = torch.clamp(
            (target_lum - self.dark_threshold) / self.dark_threshold,
            min=0.0, max=1.0
        ).detach()  # Detach is crucial to prevent "cheating"

        # 3. Scaled Loss (Critical for Balancing with L1)
        # Scale factor 0.5 maps range [0, 2] -> [0, 1]
        raw_loss = (1.0 - cosine_sim) * 0.5
        masked_loss = raw_loss * mask

        if self.reduction == 'mean':
            num_valid = torch.sum(mask) + self.eps
            return self.loss_weight * (torch.sum(masked_loss) / num_valid)
        else:
            return self.loss_weight * masked_loss

class L1ColorLoss(nn.Module):
    def __init__(self, loss_weight=1.0, color_weight=0.2, reduction='mean'):
        super(L1ColorLoss, self).__init__()
        self.loss_weight = loss_weight
        self.color_weight = color_weight  # Recommended: 0.2

        self.l1_loss = L1Loss(loss_weight=1.0, reduction=reduction)
        self.color_loss = ColorLoss(loss_weight=1.0, reduction=reduction, dark_threshold=0.03)

    def forward(self, pred, target, weight=None, **kwargs):
        l1 = self.l1_loss(pred, target, weight)
        color = self.color_loss(pred, target)
        return self.loss_weight * l1 + self.color_weight * color

# =========================================================
#  SOTA-Chasing Loss: L1 + Color + PSNR(Y-channel)
#  Use this ONLY for Phase 4 or Finetuning to break 22.8 dB
# =========================================================

class L1ColorPSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, color_weight=0.1, psnr_weight=0.02, reduction='mean', toY=True):
        super(L1ColorPSNRLoss, self).__init__()
        self.loss_weight = loss_weight
        self.color_weight = color_weight
        self.psnr_weight = psnr_weight

        # 1. Pixel Loss
        self.l1_loss = L1Loss(loss_weight=1.0, reduction=reduction)

        # 2. Color Loss (Weight lowered to 0.1 as per strategy)
        self.color_loss = ColorLoss(loss_weight=1.0, reduction=reduction, dark_threshold=0.03)

        # 3. PSNR Loss (The "Hard Key")
        # toY=True: Optimize Luminance explicitly
        self.psnr_loss = PSNRLoss(loss_weight=1.0, toY=toY)

    def forward(self, pred, target, weight=None, **kwargs):
        l1 = self.l1_loss(pred, target, weight)
        color = self.color_loss(pred, target)
        # PSNR Loss returns positive value (log error), so we add it
        psnr = self.psnr_loss(pred, target)

        return self.loss_weight * l1 + self.color_weight * color + self.psnr_weight * psnr

class L1SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, ssim_weight=0.1, reduction='mean'):
        super(L1SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = L1Loss(loss_weight=1.0, reduction=reduction)
        # 使用我们自带的可微分 _SSIMLoss (默认窗口大小11)
        self.ssim_loss = _SSIMLoss(window_size=11)

    def forward(self, pred, target, weight=None, **kwargs):
        l1 = self.l1_loss(pred, target, weight)
        # self.ssim_loss 返回的是 1 - SSIM (范围在 0 ~ 2 之间)
        ssim_loss = self.ssim_loss(pred, target)
        return self.loss_weight * l1 + self.ssim_weight * ssim_loss

# =========================================================
#  1. SSIM Loss — 可微分版本
# =========================================================
class _SSIMLoss(nn.Module):
    """
    Differentiable SSIM Loss.
    Returns 1 - SSIM (to be minimized).
    Uses avg_pool2d for efficiency, supports multi-scale.
    """
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel

    def _ssim_map(self, pred, target):
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        ws = self.window_size
        pad = ws // 2

        mu1 = F.avg_pool2d(pred, ws, stride=1, padding=pad)
        mu2 = F.avg_pool2d(target, ws, stride=1, padding=pad)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(pred * pred, ws, stride=1, padding=pad) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, ws, stride=1, padding=pad) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, ws, stride=1, padding=pad) - mu1_mu2

        # Clamp to avoid numerical issues
        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map

    def forward(self, pred, target):
        ssim_map = self._ssim_map(pred, target)
        return 1.0 - ssim_map.mean()


# =========================================================
#  2. FFT Loss — 频域约束 (幅度 + 相位)
# =========================================================
class _FFTLoss(nn.Module):
    """
    Frequency Domain Loss.
    Minimizes difference in both amplitude and phase spectra.
    Particularly effective for low-light enhancement where
    frequency separation matters (illumination=low-freq, detail=high-freq).
    """
    def __init__(self, amp_weight=1.0, phase_weight=1.0):
        super().__init__()
        self.amp_weight = amp_weight
        self.phase_weight = phase_weight

    def forward(self, pred, target):
        # 2D FFT
        pred_fft = torch.fft.rfft2(pred, norm='backward')
        target_fft = torch.fft.rfft2(target, norm='backward')

        # Amplitude loss (L1 in frequency domain)
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        amp_loss = F.l1_loss(pred_amp, target_amp)

        # Phase loss (L1 on phase angle)
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        phase_loss = F.l1_loss(pred_phase, target_phase)

        return self.amp_weight * amp_loss + self.phase_weight * phase_loss


# =========================================================
#  3. EnhancementLoss — 一体化综合损失
# =========================================================
class EnhancementLoss(nn.Module):
    """
    Combined Loss for Low-Light Image Enhancement.

    L_total = loss_weight * (
        charbonnier_weight * L_charbonnier +
        ssim_weight * L_ssim +
        color_weight * L_color +
        fft_weight * L_fft +
        psnr_weight * L_psnr
    )

    Args:
        loss_weight (float): Overall loss scaling factor. Default: 1.0.
        charbonnier_weight (float): Weight for Charbonnier (robust L1). Default: 1.0.
        ssim_weight (float): Weight for SSIM loss. Default: 0.5.
        color_weight (float): Weight for color consistency. Default: 0.3.
        fft_weight (float): Weight for FFT frequency loss. Default: 0.1.
        psnr_weight (float): Weight for PSNR loss. Default: 0.02.
        eps (float): Charbonnier epsilon. Default: 1e-3.
        toY (bool): Whether to convert to Y channel for PSNR. Default: False.
    """

    def __init__(self, loss_weight=1.0,
                 charbonnier_weight=1.0,
                 ssim_weight=0.5,
                 color_weight=0.3,
                 fft_weight=0.1,
                 psnr_weight=0.02,
                 eps=1e-3,
                 toY=False,
                 reduction='mean'):
        super(EnhancementLoss, self).__init__()
        self.loss_weight = loss_weight
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.fft_weight = fft_weight
        self.psnr_weight = psnr_weight
        self.eps = eps
        self.reduction = reduction

        # Sub-modules
        if self.ssim_weight > 0:
            self.ssim_loss = _SSIMLoss(window_size=11)

        if self.fft_weight > 0:
            self.fft_loss = _FFTLoss(amp_weight=1.0, phase_weight=0.5)

        if self.psnr_weight > 0:
            self.scale = 10 / np.log(10)
            self.toY = toY
            self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
            self.first = True

    def _charbonnier(self, pred, target):
        """Charbonnier Loss — robust L1 variant that handles outliers better."""
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

    def _color(self, pred, target, dark_threshold=0.03):
        """Color consistency via cosine similarity with dark region masking."""
        eps = 1e-6
        pred_norm = torch.norm(pred, dim=1, keepdim=True) + eps
        target_norm = torch.norm(target, dim=1, keepdim=True) + eps

        pred_vec = pred / pred_norm
        target_vec = target / target_norm

        cosine_sim = torch.sum(pred_vec * target_vec, dim=1, keepdim=True)

        # Soft mask — avoid penalizing near-black regions
        target_lum = torch.mean(target, dim=1, keepdim=True)
        mask = torch.clamp(
            (target_lum - dark_threshold) / dark_threshold,
            min=0.0, max=1.0
        ).detach()

        raw_loss = (1.0 - cosine_sim) * 0.5
        masked_loss = raw_loss * mask

        num_valid = torch.sum(mask) + eps
        return torch.sum(masked_loss) / num_valid

    def _psnr(self, pred, target):
        """PSNR Loss — directly optimizes peak signal-to-noise ratio."""
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False
            pred_y = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target_y = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            pred_y, target_y = pred_y / 255., target_y / 255.
            return self.scale * torch.log(
                ((pred_y - target_y) ** 2).mean(dim=(1, 2, 3)) + 1e-8
            ).mean()
        else:
            return self.scale * torch.log(
                ((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8
            ).mean()

    def forward(self, pred, target, weight=None, **kwargs):
        total_loss = 0.0

        # 1. Charbonnier (robust L1) — 主力像素损失
        if self.charbonnier_weight > 0:
            total_loss = total_loss + self.charbonnier_weight * self._charbonnier(pred, target)

        # 2. SSIM — 结构相似性
        if self.ssim_weight > 0:
            total_loss = total_loss + self.ssim_weight * self.ssim_loss(pred, target)

        # 3. Color — 颜色一致性
        if self.color_weight > 0:
            total_loss = total_loss + self.color_weight * self._color(pred, target)

        # 4. FFT — 频域约束
        if self.fft_weight > 0:
            total_loss = total_loss + self.fft_weight * self.fft_loss(pred, target)

        # 5. PSNR — 直接优化 PSNR
        if self.psnr_weight > 0:
            total_loss = total_loss + self.psnr_weight * self._psnr(pred, target)

        return self.loss_weight * total_loss
# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Moudle):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction

#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction


#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss
