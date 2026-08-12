import omegaconf
import torch
import torchvision

import ircgs.change_detection.base_detector as base_detector
from ircgs.utils.custom_types import Image

class DinoV2Detector(base_detector.BaseDetector):

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitb14")
        self.model.eval()
        self.model.to(self.device)
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _preprocess_image(self, image: Image) -> torch.Tensor:
        # Make image divisible by DINO patch size (14)
        DINO_PATCH_SIZE = 14
        h, w, c = image.shape
        aligned_h = h - (h % DINO_PATCH_SIZE)
        aligned_w = w - (w % DINO_PATCH_SIZE)
        # Permute to (C, H, W) for torch transforms
        image_t = image.permute(2, 0, 1)
        resize = torch.nn.functional.interpolate
        resized = resize(image_t.unsqueeze(0), size=(aligned_h, aligned_w), mode="bilinear", align_corners=False)
        # Apply ImageNet normalization - CRITICAL for DINOv2!
        normalized = self.normalize(resized.squeeze(0))
        return normalized

    def _dilate_mask(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        # mask: (H, W) or (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)
        mask = mask.float().unsqueeze(0)  # (1, 1, H, W)
        dilated = torch.nn.functional.conv2d(mask, kernel, padding=kernel_size // 2)
        dilated = (dilated > 0).float()
        return dilated.squeeze(0).squeeze(0)

    def predict_change_mask(self, rendered_image: Image, observation: Image) -> torch.Tensor:
        # Preprocess images
        rendered = self._preprocess_image(rendered_image)
        observed = self._preprocess_image(observation)
        # DINO expects (B, C, H, W)
        rendered = rendered.unsqueeze(0)
        observed = observed.unsqueeze(0)
        with torch.no_grad():
            (rendered_feats, ) = self.model.get_intermediate_layers(
                rendered, reshape=True
            )
            (observed_feats, ) = self.model.get_intermediate_layers(
                observed, reshape=True
            )
            # DINOv2 returns (B, C, H, W) for each layer after tuple unpacking
            # cos expects tensors with matching dims, apply along channel dim (dim=1)
            # NOTE: We use rendered_feats directly (not [0]) since tuple unpacking already 
            # extracted the tensor. Using [0] would incorrectly slice the batch dimension!
            cos_sim = self.cos(rendered_feats, observed_feats)  # [B, H, W]
            cos_sim = cos_sim.squeeze(0)  # Remove batch dim -> [H, W]
        
        orig_h, orig_w, _ = rendered_image.shape
        
        # IMPORTANT: Upsample cos_sim VALUES (not binary mask) using bilinear interpolation
        # This produces smoother, more accurate results than upsampling binary masks
        if getattr(self.cfg, "upsample", False):
            cos_sim = cos_sim.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            cos_sim = torch.nn.functional.interpolate(
                cos_sim, size=(orig_h, orig_w), mode='bilinear', align_corners=False
            )
            cos_sim = cos_sim.squeeze(0).squeeze(0)  # [H, W]
        
        # Threshold AFTER upsampling
        mask = cos_sim < self.cfg.threshold  # changed pixels: True if similarity below threshold
        
        # Apply dilation on full resolution
        if getattr(self.cfg, "dilate_mask", False):
            mask = self._dilate_mask(mask, kernel_size=getattr(self.cfg, "dilate_kernel_size", 31))
            mask = mask > 0.5  # Convert back to bool
        
        return mask

