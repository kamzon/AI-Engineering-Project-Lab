from typing import Dict, Iterable, List, Optional, Tuple
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from pipeline.config import ModelConstants


class FewShotResNet:
    """
    Few-shot binary fine-tuning wrapper around a pretrained image classifier.

    - Loads base model and processor from ModelConstants.IMAGE_MODEL_ID
    - Fine-tunes only the classification head (by default) to distinguish a
      user-specified target object type vs "others" using provided images.
    - Supports class weights derived from counts for imbalanced data.
    - Provides classification for SAM segments: returns target label or "others".
    """

    def __init__(
        self,
        device: Optional[str] = None,
        lr: float = ModelConstants.FEW_SHOT_LR,
        weight_decay: float = ModelConstants.FEW_SHOT_WEIGHT_DECAY,
        max_epochs: int = ModelConstants.FEW_SHOT_MAX_EPOCHS,
        batch_size: int = ModelConstants.FEW_SHOT_BATCH_SIZE,
        freeze_backbone: bool = ModelConstants.FEW_SHOT_FREEZE_BACKBONE,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.freeze_backbone = freeze_backbone

        self._model: Optional[nn.Module] = None
        self._processor: Optional[AutoImageProcessor] = None
        self._target_label: Optional[str] = None

    def _freeze_all_but_head(self, model: nn.Module) -> None:
        if not self.freeze_backbone:
            return
        for param in model.parameters():
            param.requires_grad = False
        head = getattr(model, "classifier", None) or getattr(model, "fc", None)
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True

    def _build_binary_samples(
        self,
        target_label: str,
        label_to_image_paths: Dict[str, List[str]],
        label_to_counts: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[Tuple[str, int]], torch.Tensor]:
        """
        Build samples and compute class weights.

        Positive class (1): images under `target_label`.
        Negative class (0): images under all other labels combined ("others").

        If label_to_counts provided, derive weights inversely proportional
        to positive/negative counts. Otherwise compute by number of images.
        """
        positives = label_to_image_paths.get(target_label, [])
        negatives: List[str] = []
        for label, paths in label_to_image_paths.items():
            if label != target_label:
                negatives.extend(paths)

        samples: List[Tuple[str, int]] = []
        for p in positives:
            samples.append((p, 1))
        for n in negatives:
            samples.append((n, 0))

        # Class weights: index 0 for others, index 1 for target
        if label_to_counts is not None:
            pos_count = int(label_to_counts.get(target_label, len(positives)))
            neg_count = int(sum(
                label_to_counts.get(lbl, len(paths))
                for lbl, paths in label_to_image_paths.items()
                if lbl != target_label
            ))
        else:
            pos_count = max(1, len(positives))
            neg_count = max(1, len(negatives))

        total = pos_count + neg_count
        # Inverse frequency weighting
        weight_neg = total / (2.0 * neg_count)
        weight_pos = total / (2.0 * pos_count)
        class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float)
        return samples, class_weights

    def _collate(self, batch: List[Tuple[str, int]]):
        assert self._processor is not None
        images = []
        targets = []
        for path, target in batch:
            img = Image.open(path).convert("RGB")
            images.append(img)
            targets.append(target)
        proc = self._processor(images=images, return_tensors="pt")
        pixel_values = proc["pixel_values"]
        return pixel_values, torch.tensor(targets, dtype=torch.long)

    def finetune_binary(
        self,
        target_label: str,
        label_to_image_paths: Dict[str, List[str]],
        label_to_counts: Optional[Dict[str, int]] = None,
    ) -> nn.Module:
        """
        Fine-tune the classification head to distinguish `target_label` vs "others".

        Inputs:
        - target_label: the user-selected object type
        - label_to_image_paths: mapping of label -> list of image file paths
        - label_to_counts: optional mapping of label -> integer ground-truth count to
          weight the loss (helps when datasets are imbalanced).
        """
        self._target_label = target_label
        self._processor = AutoImageProcessor.from_pretrained(ModelConstants.IMAGE_MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(
            ModelConstants.IMAGE_MODEL_ID,
            num_labels=2,
            ignore_mismatched_sizes=True,
        ).to(self.device)

        # Explicit single-label classification
        model.config.problem_type = "single_label_classification"
        model.config.label2id = {"others": 0, target_label: 1}
        model.config.id2label = {0: "others", 1: target_label}

        self._freeze_all_but_head(model)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        samples, class_weights = self._build_binary_samples(
            target_label, label_to_image_paths, label_to_counts
        )
        class_weights = class_weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model.train()
        for _ in range(self.max_epochs):
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i : i + self.batch_size]
                pixel_values, targets = self._collate(batch)
                pixel_values = pixel_values.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

        self._model = model.eval()

        # Persist artifacts for reuse
        save_dir = ModelConstants.FINETUNED_MODEL_DIR
        os.makedirs(save_dir, exist_ok=True)
        try:
            self._model.save_pretrained(save_dir)
            assert self._processor is not None
            self._processor.save_pretrained(save_dir)
        except Exception:
            pass

        return self._model

    def classify_segments(
        self,
        segments: Iterable[torch.Tensor],
        threshold: float = 0.5,
    ) -> Tuple[List[str], List[float]]:
        """
        Classify SAM-produced segments as target or "others".

        Returns two lists aligned with input order: predicted label, confidence.
        If model has 2 labels, confidence is the softmax prob of the predicted class.
        """
        assert self._model is not None
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(ModelConstants.IMAGE_MODEL_ID)

        predicted_labels: List[str] = []
        confidences: List[float] = []
        self._model.eval()
        with torch.no_grad():
            for segment in segments:
                # segment expected CHW tensor [C,H,W] in uint8 or float [0,1]
                if segment.dtype != torch.uint8:
                    segment = (segment.clamp(0, 1) * 255.0).to(torch.uint8)
                img_hwc = segment.permute(1, 2, 0).cpu().numpy()
                inputs = self._processor(images=img_hwc, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                conf, idx = probs.max(dim=-1)
                idx_int = int(idx.item())
                conf_float = float(conf.item())
                label = self._model.config.id2label.get(idx_int, "others")
                # Optional thresholding: if not confident, call it others
                if conf_float < threshold:
                    label = "others"
                predicted_labels.append(label)
                confidences.append(conf_float)
        return predicted_labels, confidences

