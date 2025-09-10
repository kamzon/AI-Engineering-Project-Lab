from typing import List, Optional

import torch
from transformers import pipeline as hf_pipeline

from pipeline.config import ModelConstants


class ZeroShotLabeler:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline = None

    def _init(self) -> None:
        if self._pipeline is None:
            device_index = 0 if (self.device == "cuda" and torch.cuda.is_available()) else -1
            self._pipeline = hf_pipeline(
                "zero-shot-classification",
                model=ModelConstants.ZERO_SHOT_MODEL_ID,
                device=device_index,
            )

    def label(self, predicted_classes: List[str], candidate_labels: List[str]) -> List[str]:
        self._init()
        labels: List[str] = []
        for predicted_class in predicted_classes:
            result = self._pipeline(predicted_class, candidate_labels=candidate_labels)
            labels.append(result["labels"][0])
        return labels


