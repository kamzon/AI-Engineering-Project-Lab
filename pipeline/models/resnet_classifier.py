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

    def _init(self) -> None:
        if self._image_processor is None or self._class_model is None:
            # Prefer a fine-tuned checkpoint if present
            finetuned_dir = ModelConstants.FINETUNED_MODEL_DIR
            load_source = (
                finetuned_dir if os.path.isdir(finetuned_dir) and os.listdir(finetuned_dir)
                else ModelConstants.IMAGE_MODEL_ID
            )
            print(f"Loading classifier from {load_source}") 
            self._image_processor = AutoImageProcessor.from_pretrained(load_source)
            self._class_model = AutoModelForImageClassification.from_pretrained(
                load_source
            ).to(self.device)

    def classify(self, segments: List[torch.Tensor]) -> Tuple[List[str], List[float]]:
        self._init()
        assert self._image_processor is not None
        assert self._class_model is not None

        predicted_classes: List[str] = []
        classifier_confidences: List[float] = []

        self._class_model.eval()
        with torch.no_grad():
            for segment in segments:
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
                except Exception:
                    classifier_confidences.append(float("nan"))

        return predicted_classes, classifier_confidences


