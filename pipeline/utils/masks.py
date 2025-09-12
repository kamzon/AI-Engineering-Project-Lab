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
    def build_panoptic_map(
        masks_sorted: List[Dict[str, Any]],
        image_size: Tuple[int, int],
        top_n: int,
        coverage_ratio: float = 0.0,
        min_rel_area: float = 0.0,
    ) -> torch.Tensor:
        width, height = image_size
        panoptic_map_np = np.zeros((height, width), dtype=np.int32)
        total_pixels = float(width * height)

        largest_area = masks_sorted[0]["area"] if masks_sorted else 0
        added = 0
        for idx, mask_data in enumerate(masks_sorted):
            if added >= top_n:
                break
            # Filter very small segments relative to the main one
            if min_rel_area > 0.0 and largest_area > 0:
                if mask_data["area"] < (min_rel_area * largest_area):
                    continue
            # Add this mask
            panoptic_map_np[mask_data["segmentation"]] = added + 1
            added += 1
            if coverage_ratio > 0.0:
                covered = float((panoptic_map_np != 0).sum()) / total_pixels
                if covered >= coverage_ratio:
                    break
        panoptic_map = torch.from_numpy(panoptic_map_np)
        return panoptic_map

    @staticmethod
    def merge_small_adjacent_segments(panoptic_map: torch.Tensor, min_ratio: float = 0.05) -> torch.Tensor:
        """
        Merge small segments into the largest segment if they touch it.

        - min_ratio: segments with area < min_ratio * area(largest) are eligible to merge.
        """
        labels = [int(l) for l in panoptic_map.unique().tolist() if int(l) != 0]
        if not labels:
            return panoptic_map
        areas = {label: int((panoptic_map == label).sum().item()) for label in labels}
        main_label = max(labels, key=lambda l: areas[l])
        main_mask = panoptic_map == main_label
        main_area = areas[main_label]
        threshold = max(1, int(main_area * float(min_ratio)))

        # 4-neighborhood adjacency check
        h, w = panoptic_map.shape
        for label in labels:
            if label == main_label:
                continue
            if areas[label] >= threshold:
                continue
            mask = (panoptic_map == label)
            # Detect if any pixel of mask touches main_mask via 4-neighborhood
            touch = False
            ys, xs = torch.nonzero(mask, as_tuple=True)
            for y, x in zip(ys.tolist(), xs.tolist()):
                if (
                    (y > 0 and main_mask[y - 1, x]) or
                    (y + 1 < h and main_mask[y + 1, x]) or
                    (x > 0 and main_mask[y, x - 1]) or
                    (x + 1 < w and main_mask[y, x + 1])
                ):
                    touch = True
                    break
            if touch:
                panoptic_map[mask] = main_label
                main_mask = panoptic_map == main_label
                main_area += areas[label]
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


