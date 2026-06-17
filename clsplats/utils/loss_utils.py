"""
Loss functions for Gaussian Splatting training.
"""

import torch
import torch.nn.functional as F
from math import exp


def l1_loss(network_output, gt):
    """L1 loss between network output and ground truth."""
    return torch.abs(network_output - gt).mean()


def l2_loss(network_output, gt):
    """L2 loss between network output and ground truth."""
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    """Generate 1D Gaussian kernel."""
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create 2D Gaussian window for SSIM."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    """
    Compute SSIM between two images.
    
    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        window_size: Size of Gaussian window
        size_average: Whether to average over spatial dimensions
        
    Returns:
        SSIM value
    """
    channel = img1.size(1)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Internal SSIM computation."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def combined_loss(rendered, target, lambda_dssim=0.2):
    """
    Combined L1 + SSIM loss.
    
    Args:
        rendered: Rendered image [B, C, H, W] or [C, H, W]
        target: Target image [B, C, H, W] or [C, H, W]
        lambda_dssim: Weight for SSIM loss
        
    Returns:
        Combined loss value
    """
    # Ensure 4D tensors
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    
    l1 = l1_loss(rendered, target)
    ssim_val = ssim(rendered, target)
    
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_val)
