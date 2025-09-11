from typing import Dict, List, Optional, Tuple
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from pipeline.config import ModelConstants


class FewShotResNet:
    def __init__(
        self,
        device: Optional[str] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        max_epochs: int = 3,
        batch_size: int = 8,
        freeze_backbone: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.freeze_backbone = freeze_backbone

        self._model: Optional[nn.Module] = None
        self._processor: Optional[AutoImageProcessor] = None

    def _build_dataset(self, class_to_image_paths: Dict[str, List[str]]):
        samples: List[Tuple[str, int]] = []
        classes = sorted(class_to_image_paths.keys())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls, paths in class_to_image_paths.items():
            idx = class_to_idx[cls]
            for p in paths:
                samples.append((p, idx))
        return samples, class_to_idx

    def _collate(self, batch: List[Tuple[str, int]]):
        assert self._processor is not None
        images = []
        targets = []
        for path, idx in batch:
            img = Image.open(path).convert("RGB")
            images.append(img)
            targets.append(idx)
        proc = self._processor(images=images, return_tensors="pt")
        pixel_values = proc["pixel_values"]
        return pixel_values, torch.tensor(targets, dtype=torch.long)

    def _freeze_backbone_except_head(self, model: nn.Module) -> None:
        if not self.freeze_backbone:
            return
        for p in model.parameters():
            p.requires_grad = False
        head = getattr(model, "classifier", None)
        if head is None:
            head = getattr(model, "fc", None)
        if head is not None:
            for p in head.parameters():
                p.requires_grad = True

    def finetune(self, class_to_image_paths: Dict[str, List[str]]) -> nn.Module:
        print(f"Finetuning with {len(class_to_image_paths)} classes")
        samples, class_to_idx = self._build_dataset(class_to_image_paths)
        num_classes = len(class_to_idx)
        print(f"Number of classes: {num_classes}")
        self._processor = AutoImageProcessor.from_pretrained(ModelConstants.IMAGE_MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(
            ModelConstants.IMAGE_MODEL_ID,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        ).to(self.device)

        model.config.label2id = {c: i for c, i in class_to_idx.items()}
        model.config.id2label = {i: c for c, i in class_to_idx.items()}

        self._freeze_backbone_except_head(model)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        model.train()
        for epoch in range(self.max_epochs):
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i : i + self.batch_size]
                pixel_values, targets = self._collate(batch)
                pixel_values = pixel_values.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(pixel_values=pixel_values, labels=targets)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        # Switch to eval and persist the model + processor
        self._model = model.eval()

        # Save fine-tuned weights and config to a directory for later reuse
        save_dir = ModelConstants.FINETUNED_MODEL_DIR
        os.makedirs(save_dir, exist_ok=True)
        try:
            self._model.save_pretrained(save_dir)
            assert self._processor is not None
            self._processor.save_pretrained(save_dir)
        except Exception:
            # If saving fails, still return the model for immediate use
            pass

        return self._model

    def classify(self, segments: List[torch.Tensor]) -> Tuple[List[str], List[float]]:
        assert self._model is not None
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(ModelConstants.IMAGE_MODEL_ID)
        predicted_classes: List[str] = []
        classifier_confidences: List[float] = []
        self._model.eval()
        with torch.no_grad():
            for segment in segments:
                img_hwc = segment.permute(1, 2, 0).cpu().numpy().astype("uint8")
                inputs = self._processor(images=img_hwc, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                logits = outputs.logits
                predicted_class_idx = int(logits.argmax(-1).item())
                predicted_class = self._model.config.id2label[predicted_class_idx]
                predicted_classes.append(predicted_class)
                try:
                    probs = torch.softmax(logits, dim=-1)
                    conf = float(probs.max(dim=-1).values.item())
                    classifier_confidences.append(conf)
                except Exception:
                    classifier_confidences.append(float("nan"))
        return predicted_classes, classifier_confidences


