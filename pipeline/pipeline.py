import os
import uuid
import time
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
import torchvision.transforms as tf

from pipeline.metrics import MetricsCollector
from pipeline.config import ModelConstants
from pipeline.models.safety import load_safety_model
from pipeline.models.grounded_sam2 import GroundedSAM2
from pipeline.models.resnet_classifier import ResNetImageClassifier
from pipeline.models.zero_shot import ZeroShotLabeler
from pipeline.utils.visualization import PanopticVisualizer
from pipeline.utils.image_utils import ImageUtils
from pipeline.utils.image_utils import ImageUtils


class Pipeline:

    def __init__(
        self,
        image_path: str = "pipeline/inputs/image.png",
        background_fill: int = 188,
        device: Optional[str] = None,
        use_finetuned_classifier: Optional[bool] = None,
    ) -> None:
        self.image_path = image_path
        self.background_fill = background_fill
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.candidate_labels = ModelConstants.DEFAULT_CANDIDATE_LABELS
        # None means auto (prefer finetuned if present)
        self.use_finetuned_classifier: Optional[bool] = use_finetuned_classifier

        self._original_image: Optional[Image.Image] = None
        self._image: Optional[Image.Image] = None
        self._detections: Optional[List[Dict[str, Any]]] = None
        self._segments: Optional[List[torch.Tensor]] = None
        self._predicted_classes: Optional[List[str]] = None
        self._zero_shot_labels: Optional[List[str]] = None
        self._classifier_confidences: Optional[List[float]] = None
        self.image_path = image_path
        self.safety_model = load_safety_model(ModelConstants.SAFETY_MODEL_PATH, self.device)  
        self._overlay_path: Optional[str] = None


    
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
        """Step 1: Detect objects using GroundedSAM2"""
        t0 = time.perf_counter()
        print(f"[Pipeline] candidate_labels={self.candidate_labels}")
        # Query all candidate labels to maximize recall
        detection_queries: List[str] = list(self.candidate_labels) if self.candidate_labels else []
        print(f"[Pipeline] detection_queries={detection_queries}")
        model = GroundedSAM2(device=self.device)
        detections = model.detect(
            image=self._image,
            text_queries=detection_queries,
            box_threshold=ModelConstants.GROUNDING_BOX_THRESHOLD,
            text_threshold=ModelConstants.GROUNDING_TEXT_THRESHOLD,
        )
        print(f"[Pipeline] detections_count={len(detections)}")
        self._detections = detections
        
        # Crop segments from detections for ResNet classification
        self._segments = ImageUtils.crop_segments_from_detections(
            image=self._image,
            detections=detections,
            background_fill=self.background_fill,
            min_size=32,
        )
        print(f"[Pipeline] segments_cropped={len(self._segments)}")
        
        return (time.perf_counter() - t0) * 1000.0
    
    def _step_classifier(self) -> float:
        """Step 2: Classify cropped segments using ResNet"""
        t0 = time.perf_counter()
        assert self._segments is not None
        classifier = ResNetImageClassifier(device=self.device, use_finetuned=self.use_finetuned_classifier)
        self._predicted_classes, self._classifier_confidences = classifier.classify(
            self._segments
        )
        print(f"[Pipeline] resnet_classes_count={len(self._predicted_classes)} by_label={ {l: self._predicted_classes.count(l) for l in set(self._predicted_classes)} }")
        return (time.perf_counter() - t0) * 1000.0
    
    def _step_zero_shot(self) -> float:
        """Step 3: Group/refine labels using ZeroShotLabeler"""
        t0 = time.perf_counter()
        assert self._predicted_classes is not None
        labeler = ZeroShotLabeler(device=self.device)
        self._zero_shot_labels = labeler.label(
            self._predicted_classes, self.candidate_labels
        )
        print(f"[Pipeline] zero_shot_labels_count={len(self._zero_shot_labels)} by_label={ {l: self._zero_shot_labels.count(l) for l in set(self._zero_shot_labels)} }")
        return (time.perf_counter() - t0) * 1000.0
    
    def _step_save_overlay(self, run_id: str) -> str:
        """Save detection overlay for frontend"""
        overlay_path = os.path.join("pipeline", "outputs", f"{run_id}_overlay.png")
        # Attach ResNet labels/confidences to detections for visualization
        detections_with_labels: List[Dict[str, Any]] = list(self._detections or [])
        if detections_with_labels and self._predicted_classes is not None:
            for i, det in enumerate(detections_with_labels):
                try:
                    det["resnet_label"] = self._predicted_classes[i]
                    if self._classifier_confidences is not None and i < len(self._classifier_confidences):
                        det["resnet_conf"] = float(self._classifier_confidences[i])
                except Exception:
                    # Best-effort enrichment; continue on any mismatch
                    pass
        PanopticVisualizer().save_detections(self._image, detections_with_labels, overlay_path)
        self._overlay_path = overlay_path
        return overlay_path

    def _build_models_used(self) -> Dict[str, Any]:
        # Report whether a fine-tuned classifier was used
        classifier_source = (
            ModelConstants.FINETUNED_MODEL_DIR
            if os.path.isdir(ModelConstants.FINETUNED_MODEL_DIR)
            and os.listdir(ModelConstants.FINETUNED_MODEL_DIR)
            else ModelConstants.IMAGE_MODEL_ID
        )
        return {
            "grounded": {
                "dino_model": getattr(ModelConstants, "GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-base"),
            },
            "classifier": {"id": classifier_source},
            "zero_shot": {"id": ModelConstants.ZERO_SHOT_MODEL_ID},
        }

    def _build_label_counts(self) -> Dict[str, int]:
        labels = self._zero_shot_labels or self._predicted_classes or []
        counts = {label: sum(1 for l in labels if l == label) for label in self.candidate_labels}
        print(f"[Pipeline] label_counts={counts}")
        return counts

    def _step_metrics(self, safety_ms: float, grounded_ms: float, classifier_ms: float, zero_shot_ms: float, overall_t0: float, safety_info: Dict[str, Any]) -> Dict[str, Any]:
        metrics_collector = MetricsCollector()
        metadata = metrics_collector.build(
            image=self._original_image,
            segments=self._detections,  # treated as segments for counting/area by metrics
            zero_shot_labels=self._zero_shot_labels,
            predicted_classes=self._predicted_classes,
            models_used=self._build_models_used(),
            classifier_confidences=self._classifier_confidences,
            timings_ms={
                "safety_ms": safety_ms,
                "sam_ms": grounded_ms,
                "classifier_ms": classifier_ms,
                "zero_shot_ms": zero_shot_ms,
                "overall_ms": (time.perf_counter() - overall_t0) * 1000.0,
            },
            safety=safety_info,
        )
        return metadata
    
    def _safety_check(self) -> (bool, Dict[str, Any], float):
        print("Running safety filter...")
        t0 = time.perf_counter()
        transform = tf.Compose([
            tf.Resize((224, 224)),
            tf.ToTensor()
        ])
        image = transform(self._image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.safety_model(image)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0, pred].item()

            # Assume class index 1 corresponds to 'unsafe' and 0 to 'safe'
            label = "unsafe" if pred == 1 else "safe"
            print(f"Safety filter result: {label.upper()} (conf {confidence:.2f})")
            
            # Only deny if predicted as unsafe AND confidence > threshold
            threshold = ModelConstants.SAFETY_CONFIDENCE_THRESHOLD
            allowed = not (pred == 1 and confidence > threshold)
            if not allowed:
                print(f"Image DENIED: unsafe prediction with high confidence ({confidence:.2f} > {threshold})")
            else:
                print(f"Image ALLOWED: {label} prediction with confidence {confidence:.2f}")

            safety_info = {
                "label": label,
                "pred_index": pred,
                "confidence": float(confidence),
                "threshold": float(threshold),
            }
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return allowed, safety_info, elapsed_ms


    def run(self) -> Dict[str, Any]:
        # Load and preprocess image once
        self._original_image = Image.open(self.image_path).convert("RGB")
        self._image = ImageUtils.resize_longest_side(
            self._original_image, ModelConstants.IMAGE_LONGEST_SIDE
        )

        # safety check
        allowed, safety_info, safety_ms = self._safety_check()
        if not allowed:
            print("Image flagged as UNSAFE. Aborting pipeline.")
            # Build minimal metrics metadata even on unsafe exit
            grounded_ms = 0.0
            classifier_ms = 0.0
            zero_shot_ms = 0.0
            # Measure overall as just the safety time in this early-exit scenario
            overall_t0 = time.perf_counter() - (safety_ms / 1000.0)
            metadata = self._step_metrics(
                safety_ms,
                grounded_ms,
                classifier_ms,
                zero_shot_ms,
                overall_t0,
                safety_info,
            )
            return {
                "error": "Image classified as UNSAFE. Aborting pipeline.",
                "metadata": metadata,
                "label_counts": self._build_label_counts(),
            }

        print("Image passed safety filter. Continuing pipeline...")
        overall_t0 = time.perf_counter()
        
        # Step 1: Object detection with GroundedSAM2
        grounded_ms = self._step_grounded()
        
        # Step 2: ResNet classification on cropped segments
        classifier_ms = self._step_classifier()
        
        # Step 3: ZeroShot label grouping/refinement
        zero_shot_ms = self._step_zero_shot()
        
        # Save overlay visualization
        run_id = uuid.uuid4().hex
        self._step_save_overlay(run_id)
        
        # Collect metrics
        metadata = self._step_metrics(safety_ms, grounded_ms, classifier_ms, zero_shot_ms, overall_t0, safety_info)
        
        result: Dict[str, Any] = {
            "predicted_classes": self._predicted_classes,
            "zero_shot_labels": self._zero_shot_labels,
            "label_counts": self._build_label_counts(),
            "id": run_id,
            "panoptic_path": self._overlay_path,
            "metadata": metadata,
            "detections": self._detections,
            "finetuned_available": bool(os.path.isdir(ModelConstants.FINETUNED_MODEL_DIR) and os.listdir(ModelConstants.FINETUNED_MODEL_DIR)),
            "using_finetuned": bool(self.use_finetuned_classifier) if self.use_finetuned_classifier is not None else bool(os.path.isdir(ModelConstants.FINETUNED_MODEL_DIR) and os.listdir(ModelConstants.FINETUNED_MODEL_DIR)),
        }
        return result


if __name__ == "__main__":
    pipeline_runner = Pipeline()
    pipeline_runner.image_path = "pipeline/inputs/image2.jpg"
    print(pipeline_runner.run())
