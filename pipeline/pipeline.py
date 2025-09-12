import os
import uuid
import time
from typing import Any, Dict, List, Optional

import torch
import torchvision.transforms as tf
from PIL import Image

from pipeline.metrics import MetricsCollector
from pipeline.config import ModelConstants
from pipeline.models.resnet_classifier import ResNetImageClassifier
from pipeline.models.zero_shot import ZeroShotLabeler
from pipeline.utils.masks import SamMaskUtils
from pipeline.utils.visualization import PanopticVisualizer


class Pipeline:

    def __init__(
        self,
        image_path: str = "pipeline/inputs/image.png",
        top_n: int = 10,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.85,
        min_mask_region_area: int = 500,
        background_fill: int = 188,
        device: Optional[str] = None,
    ) -> None:
        self.image_path = image_path
        self.top_n = top_n
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self.background_fill = background_fill
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.candidate_labels = ModelConstants.DEFAULT_CANDIDATE_LABELS

        self._original_image: Optional[Image.Image] = None
        self._image: Optional[Image.Image] = None
        self._image_tensor: Optional[torch.Tensor] = None
        self._panoptic_map: Optional[torch.Tensor] = None
        self._segments: Optional[List[torch.Tensor]] = None
        self._predicted_classes: Optional[List[str]] = None
        self._zero_shot_labels: Optional[List[str]] = None
        self._classifier_confidences: Optional[List[float]] = None


    def _load_image(self) -> Image.Image:
        if self._image is None:
            self._original_image = Image.open(self.image_path)
            image = self._original_image.convert("RGB")
            target_longest_side = ModelConstants.IMAGE_LONGEST_SIDE
            width, height = image.size
            longest_side = max(width, height)
            if longest_side != target_longest_side:
                scale = target_longest_side / float(longest_side)
                new_width = max(1, int(round(width * scale)))
                new_height = max(1, int(round(height * scale)))
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = (
                        getattr(Image, "LANCZOS", None)
                        or getattr(Image, "BICUBIC", 0)
                    )
                image = image.resize(
                    (new_width, new_height), resample=resample)
            self._image = image
        return self._image

    def _to_image_tensor(self) -> torch.Tensor:
        if self._image_tensor is None:
            transform = tf.Compose([tf.PILToTensor()])
            self._image_tensor = transform(self._load_image())
        return self._image_tensor



    def correct_predictions(
        self,
        classes: Optional[Dict[int, str]] = None,
        labels: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        if self._segments is None:
            raise ValueError("No segments available. Run the pipeline first.")

        if self._predicted_classes is None:
            # If predictions were not computed via run(), initialize placeholders
            self._predicted_classes = ["" for _ in range(len(self._segments))]
        if self._zero_shot_labels is None:
            self._zero_shot_labels = ["" for _ in range(len(self._segments))]

        if classes:
            for idx, value in classes.items():
                if idx < 0 or idx >= len(self._segments):
                    raise IndexError(f"Segment index out of range: {idx}")
                self._predicted_classes[idx] = value

        if labels:
            for idx, value in labels.items():
                if idx < 0 or idx >= len(self._segments):
                    raise IndexError(f"Segment index out of range: {idx}")
                self._zero_shot_labels[idx] = value

        return {
            "image": self._image,
            "panoptic_map": self._panoptic_map,
            "segments": self._segments,
            "predicted_classes": self._predicted_classes,
            "zero_shot_labels": self._zero_shot_labels,
        }

    def _step_sam(self) -> float:
        t0 = time.perf_counter()
        sam_utils = SamMaskUtils(
            device=self.device,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )
        mask_generator = sam_utils.init_generator()
        masks_sorted = SamMaskUtils.generate_sorted_masks(
            self._load_image(), mask_generator)
        # Build panoptic with adaptive controls:
        # - keep at most top_n segments
        # - stop early if coverage reaches 80%
        # - drop segments smaller than 2% of the largest one
        self._panoptic_map = SamMaskUtils.build_panoptic_map(
            masks_sorted,
            self._load_image().size,
            self.top_n,
            coverage_ratio=0.8,
            min_rel_area=0.02,
        )
        # Merge small adjacent segments into the main object to reduce over-segmentation
        self._panoptic_map = SamMaskUtils.merge_small_adjacent_segments(
            self._panoptic_map, min_ratio=0.05
        )
        img_tensor = self._to_image_tensor()
        self._segments = SamMaskUtils.crop_segments(
            img_tensor, self._panoptic_map, self.background_fill)
        return (time.perf_counter() - t0) * 1000.0

    def _step_classifier(self) -> float:
        t0 = time.perf_counter()
        assert self._segments is not None
        classifier = ResNetImageClassifier(device=self.device)
        self._predicted_classes, self._classifier_confidences = classifier.classify(
            self._segments
        )
        return (time.perf_counter() - t0) * 1000.0

    def _step_zero_shot(self, do_zero_shot: bool) -> float:
        zero_shot_ms = 0.0
        if do_zero_shot:
            t0 = time.perf_counter()
            labeler = ZeroShotLabeler(device=self.device)
            assert self._predicted_classes is not None
            self._zero_shot_labels = labeler.label(
                self._predicted_classes, self.candidate_labels
            )
            zero_shot_ms = (time.perf_counter() - t0) * 1000.0
        return zero_shot_ms

    def _step_save_panoptic(self, run_id: str) -> str:
        panoptic_path = os.path.join("pipeline", "outputs", f"{run_id}.png")
        assert self._panoptic_map is not None
        PanopticVisualizer().save(
            self._panoptic_map, panoptic_path, labels=self._zero_shot_labels, annotate=True)
        return panoptic_path

    def _build_models_used(self) -> Dict[str, Any]:
        # Report whether a fine-tuned classifier was used
        classifier_source = (
            ModelConstants.FINETUNED_MODEL_DIR
            if os.path.isdir(ModelConstants.FINETUNED_MODEL_DIR)
            and os.listdir(ModelConstants.FINETUNED_MODEL_DIR)
            else ModelConstants.IMAGE_MODEL_ID
        )
        return {
            "sam": {
                "variant": ModelConstants.SAM_VARIANT,
                "checkpoint": ModelConstants.SAM_CHECKPOINT_FILENAME,
            },
            "classifier": {"id": classifier_source},
            "zero_shot": {"id": ModelConstants.ZERO_SHOT_MODEL_ID},
        }

    def _build_label_counts(self) -> Dict[str, int]:
        return {
            label: sum(1 for l in (
                self._zero_shot_labels or []) if l == label)
            for label in self.candidate_labels
        }

    def _step_metrics(self, sam_ms: float, classifier_ms: float, zero_shot_ms: float, overall_t0: float) -> Dict[str, Any]:
        metrics_collector = MetricsCollector()
        metadata = metrics_collector.build(
            image=self._original_image,
            segments=self._segments,
            zero_shot_labels=self._zero_shot_labels,
            predicted_classes=self._predicted_classes,
            models_used=self._build_models_used(),
            classifier_confidences=self._classifier_confidences,
            timings_ms={
                "sam_ms": sam_ms,
                "classifier_ms": classifier_ms,
                "zero_shot_ms": zero_shot_ms,
                "overall_ms": (time.perf_counter() - overall_t0) * 1000.0,
            },
        )
        return metadata

    def run(self, do_zero_shot: bool = True) -> Dict[str, Any]:
        overall_t0 = time.perf_counter()
        sam_ms = self._step_sam()
        classifier_ms = self._step_classifier()
        zero_shot_ms = self._step_zero_shot(do_zero_shot)
        run_id = uuid.uuid4().hex
        panoptic_path = self._step_save_panoptic(run_id)
        metadata = self._step_metrics(
            sam_ms, classifier_ms, zero_shot_ms, overall_t0)
        result: Dict[str, Any] = {
            "predicted_classes": self._predicted_classes,
            "zero_shot_labels": self._zero_shot_labels,
            "label_counts": self._build_label_counts(),
            "id": run_id,
            "panoptic_path": panoptic_path,
            "metadata": metadata,
        }
        return result


if __name__ == "__main__":
    pipeline_runner = Pipeline()
    pipeline_runner.image_path = "pipeline/inputs/image2.jpg"
    print(pipeline_runner.run())
