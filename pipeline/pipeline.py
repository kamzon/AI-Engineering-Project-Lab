import os
import uuid
import time
from typing import Any, Dict, List, Optional

import torch
import torchvision.transforms as tf
from PIL import Image

from pipeline.metrics import MetricsCollector
from pipeline.config import ModelConstants
from pipeline.models.safety import load_safety_model
from pipeline.models.grounded_sam2 import GroundedSAM2
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
        self._detections: Optional[List[Dict[str, Any]]] = None
        self._predicted_classes: Optional[List[str]] = None
        self._zero_shot_labels: Optional[List[str]] = None
        self._classifier_confidences: Optional[List[float]] = None
        self.image_path = image_path
        self.safety_model = load_safety_model(ModelConstants.SAFETY_MODEL_PATH, self.device)  
        self._overlay_path: Optional[str] = None


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
        if self._detections is None:
            raise ValueError("No detections available. Run the pipeline first.")

        if self._predicted_classes is None:
            self._predicted_classes = [d.get("label", "") for d in self._detections]
        if self._zero_shot_labels is None:
            self._zero_shot_labels = list(self._predicted_classes)

        if classes:
            for idx, value in classes.items():
                if idx < 0 or idx >= len(self._detections):
                    raise IndexError(f"Detection index out of range: {idx}")
                self._predicted_classes[idx] = value

        if labels:
            for idx, value in labels.items():
                if idx < 0 or idx >= len(self._detections):
                    raise IndexError(f"Detection index out of range: {idx}")
                self._zero_shot_labels[idx] = value

        return {
            "image": self._image,
            "detections": self._detections,
            "predicted_classes": self._predicted_classes,
            "zero_shot_labels": self._zero_shot_labels,
        }

    def _step_grounded(self) -> float:
        t0 = time.perf_counter()
        print(f"[Pipeline] candidate_labels={self.candidate_labels}")
        # Use only the selected object type (first candidate) for detection
        detection_queries: List[str] = [self.candidate_labels[0]] if self.candidate_labels else []
        print(f"[Pipeline] detection_queries={detection_queries}")
        model = GroundedSAM2(device=self.device)
        detections = model.detect(
            image=self._load_image(),
            text_queries=detection_queries,
            box_threshold=ModelConstants.GROUNDING_BOX_THRESHOLD,
            text_threshold=ModelConstants.GROUNDING_TEXT_THRESHOLD,
        )
        print(f"[Pipeline] detections_count={len(detections)}")
        self._detections = detections
        # Derive classes directly from detection labels
        self._predicted_classes = [d.get("label", "") for d in detections]
        self._zero_shot_labels = list(self._predicted_classes)
        print(f"[Pipeline] predicted_classes_count={len(self._predicted_classes)} by_label={ {l: self._predicted_classes.count(l) for l in set(self._predicted_classes)} }")
        # Save detection overlay for frontend
        run_id = uuid.uuid4().hex
        overlay_path = os.path.join("pipeline", "outputs", f"{run_id}_overlay.png")
        PanopticVisualizer().save_detections(self._load_image(), detections, overlay_path)
        self._overlay_path = overlay_path
        return (time.perf_counter() - t0) * 1000.0

    def _build_models_used(self) -> Dict[str, Any]:
        return {
            "grounded": {
                "dino_model": getattr(ModelConstants, "GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-base"),
            }
        }

    def _build_label_counts(self) -> Dict[str, int]:
        labels = self._zero_shot_labels or self._predicted_classes or []
        counts = {label: sum(1 for l in labels if l == label) for label in self.candidate_labels}
        print(f"[Pipeline] label_counts={counts}")
        return counts

    def _step_metrics(self, grounded_ms: float, overall_t0: float) -> Dict[str, Any]:
        metrics_collector = MetricsCollector()
        metadata = metrics_collector.build(
            image=self._original_image,
            segments=self._detections,  # treated as segments for counting/area by metrics
            zero_shot_labels=self._zero_shot_labels,
            predicted_classes=self._predicted_classes,
            models_used=self._build_models_used(),
            classifier_confidences=self._classifier_confidences,
            timings_ms={
                "sam_ms": grounded_ms,
                "classifier_ms": 0.0,
                "zero_shot_ms": 0.0,
                "overall_ms": (time.perf_counter() - overall_t0) * 1000.0,
            },
        )
        return metadata
    
    def _safety_check(self) -> bool:
        print("Running safety filter...")
        transform = tf.Compose([
            tf.Resize((224, 224)),
            tf.ToTensor()
        ])
        image = transform(self._load_image()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.safety_model(image)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0, pred].item()

            # Assume class index 1 corresponds to 'unsafe' and 0 to 'safe'
            label = "unsafe" if pred == 1 else "safe"
            print(f"Safety filter result: {label.upper()} (conf {confidence:.2f})")
            # Return True if SAFE, False if UNSAFE
            return pred == 0 


    def run(self) -> Dict[str, Any]:
        # safety check
        if not self._safety_check():
            print("Image flagged as UNSAFE. Aborting pipeline.")
            return {"error": "Image classified as UNSAFE. Aborting pipeline."}

        print("Image passed safety filter. Continuing pipeline...")
        overall_t0 = time.perf_counter()
        grounded_ms = self._step_grounded()
        metadata = self._step_metrics(grounded_ms, overall_t0)
        result: Dict[str, Any] = {
            "predicted_classes": self._predicted_classes,
            "zero_shot_labels": self._zero_shot_labels,
            "label_counts": self._build_label_counts(),
            "id": uuid.uuid4().hex,
            "panoptic_path": self._overlay_path,
            "metadata": metadata,
            "detections": self._detections,
        }
        return result


if __name__ == "__main__":
    pipeline_runner = Pipeline()
    pipeline_runner.image_path = "pipeline/inputs/image2.jpg"
    print(pipeline_runner.run())
