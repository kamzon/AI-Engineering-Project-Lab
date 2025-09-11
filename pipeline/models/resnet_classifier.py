from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from pipeline.config import ModelConstants
import os


class ResNetImageClassifier:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._image_processor: Optional[AutoImageProcessor] = None
        self._class_model: Optional[AutoModelForImageClassification] = None
        self._load_source: Optional[str] = None

    def _init(self) -> None:
        if self._image_processor is None or self._class_model is None:
            # Prefer a fine-tuned checkpoint if present
            finetuned_dir = ModelConstants.FINETUNED_MODEL_DIR
            load_source = (
                finetuned_dir if os.path.isdir(finetuned_dir) and os.listdir(finetuned_dir)
                else ModelConstants.IMAGE_MODEL_ID
            )
            self._load_source = load_source
            print(f"Loading classifier from {load_source}") 
            if load_source == finetuned_dir:
                try:
                    print(f"Fine-tuned directory contents: {os.listdir(finetuned_dir)}")
                except Exception as e:
                    print(f"Unable to list fine-tuned directory '{finetuned_dir}': {e}")
            self._image_processor = AutoImageProcessor.from_pretrained(load_source)
            self._class_model = AutoModelForImageClassification.from_pretrained(
                load_source
            ).to(self.device)
            try:
                id2label = getattr(self._class_model.config, "id2label", {})
                problem_type = getattr(self._class_model.config, "problem_type", None)
                print(f"Classifier config id2label: {id2label}")
                print(f"Classifier config problem_type: {problem_type}")
            except Exception:
                pass

    def classify(self, segments: List[torch.Tensor]) -> Tuple[List[str], List[float]]:
        self._init()
        assert self._image_processor is not None
        assert self._class_model is not None
        if self._load_source is not None:
            print(f"Running classification using model from {self._load_source}")

        predicted_classes: List[str] = []
        classifier_confidences: List[float] = []

        self._class_model.eval()
        with torch.no_grad():
            for idx, segment in enumerate(segments):
                img_hwc = (
                    segment.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                )
                inputs = self._image_processor(images=img_hwc, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self._class_model(**inputs)
                logits = outputs.logits
                predicted_class_idx = int(logits.argmax(-1).item())
                predicted_class = self._class_model.config.id2label[predicted_class_idx]
                predicted_classes.append(predicted_class)
                try:
                    probs = torch.softmax(logits, dim=-1)
                    conf = float(probs.max(dim=-1).values.item())
                    classifier_confidences.append(conf)
                    print(f"[ResNet] Segment {idx}: confidence={conf:.4f}, class='{predicted_class}'")
                except Exception:
                    classifier_confidences.append(float("nan"))
                    print(f"[ResNet] Segment {idx}: confidence=nan, class='{predicted_class}'")

        return predicted_classes, classifier_confidences


