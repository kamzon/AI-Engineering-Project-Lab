from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from pipeline.config import ModelConstants


class SamMaskUtils:
    def __init__(self, device: str, points_per_side: int, pred_iou_thresh: float, stability_score_thresh: float, min_mask_region_area: int) -> None:
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self._mask_generator: Optional[SamAutomaticMaskGenerator] = None

    def _ensure_sam_checkpoint(self) -> None:
        import os
        import urllib.request

        if not os.path.exists(ModelConstants.SAM_CHECKPOINT_FILENAME):
            urllib.request.urlretrieve(
                ModelConstants.SAM_CHECKPOINT_URL,
                ModelConstants.SAM_CHECKPOINT_FILENAME,
            )

    def init_generator(self) -> SamAutomaticMaskGenerator:
        if self._mask_generator is None:
            self._ensure_sam_checkpoint()
            sam_model = sam_model_registry[ModelConstants.SAM_VARIANT](
                checkpoint=ModelConstants.SAM_CHECKPOINT_FILENAME
            )
            sam_model.to(self.device)
            self._mask_generator = SamAutomaticMaskGenerator(
                model=sam_model,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                min_mask_region_area=self.min_mask_region_area,
            )
        return self._mask_generator

    @staticmethod
    def generate_sorted_masks(image: Image.Image, mask_generator: SamAutomaticMaskGenerator) -> List[Dict[str, Any]]:
        image_np = np.array(image)
        masks = mask_generator.generate(image_np)
        masks_sorted = sorted(masks, key=lambda x: x["area"], reverse=True)
        return masks_sorted

    @staticmethod
    def build_panoptic_map(masks_sorted: List[Dict[str, Any]], image_size: Tuple[int, int], top_n: int) -> torch.Tensor:
        width, height = image_size
        panoptic_map_np = np.zeros((height, width), dtype=np.int32)
        for idx, mask_data in enumerate(masks_sorted[: top_n]):
            panoptic_map_np[mask_data["segmentation"]] = idx + 1
        panoptic_map = torch.from_numpy(panoptic_map_np)
        return panoptic_map

    @staticmethod
    def _get_mask_box(tensor: torch.Tensor) -> Tuple[Optional[int], Optional[int]]:
        non_zero_indices = torch.nonzero(tensor, as_tuple=True)[0]
        if non_zero_indices.numel() == 0:
            return None, None
        first_n = non_zero_indices[:1].item()
        last_n = non_zero_indices[-1:].item()
        return first_n, last_n

    @classmethod
    def crop_segments(cls, image_tensor: torch.Tensor, panoptic_map: torch.Tensor, background_fill: int) -> List[torch.Tensor]:
        segments: List[torch.Tensor] = []
        labels = [int(l) for l in panoptic_map.unique().tolist() if int(l) != 0]
        for label in labels:
            mask = panoptic_map == label
            y_start, y_end = cls._get_mask_box(mask)
            x_start, x_end = cls._get_mask_box(mask.T)
            if None in (y_start, y_end, x_start, x_end):
                continue
            cropped_tensor = image_tensor[:, y_start : y_end + 1, x_start : x_end + 1]
            cropped_mask = mask[y_start : y_end + 1, x_start : x_end + 1]
            segment = cropped_tensor * cropped_mask.unsqueeze(0)
            segment[:, ~cropped_mask] = background_fill
            segments.append(segment)
        return segments


