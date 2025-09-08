import os
import uuid
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image, ImageDraw, ImageFont

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    pipeline as hf_pipeline,
)

# Local metrics collector
try:
    from .metrics import MetricsCollector  # type: ignore
except Exception:  # pragma: no cover - fallback for direct script execution
    from metrics import MetricsCollector  # type: ignore


class ModelConstants:
    # SAM
    SAM_VARIANT: str = "vit_b"
    SAM_CHECKPOINT_FILENAME: str = "sam_vit_b_01ec64.pth"
    SAM_CHECKPOINT_URL: str = (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    )

    # Image classification
    IMAGE_MODEL_ID: str = "microsoft/resnet-50"

    # Image resizing
    IMAGE_LONGEST_SIDE: int = 1024

    DEFAULT_CANDIDATE_LABELS: List[str] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
        "building", "road", "sky", "ground", "water"
    ]

    # Zero-shot classification
    ZERO_SHOT_MODEL_ID: str = "typeform/distilbert-base-uncased-mnli"


class SamSegmentationClassifier:

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
        show_plots: bool = False,
    ) -> None:
        self.image_path = image_path
        self.top_n = top_n
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self.background_fill = background_fill
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.show_plots = show_plots
        self.candidate_labels = ModelConstants.DEFAULT_CANDIDATE_LABELS

        self._original_image: Optional[Image.Image] = None
        self._image: Optional[Image.Image] = None
        self._image_tensor: Optional[torch.Tensor] = None
        self._sam_model = None
        self._mask_generator: Optional[SamAutomaticMaskGenerator] = None
        self._masks_sorted: Optional[List[Dict[str, Any]]] = None
        self._panoptic_map: Optional[torch.Tensor] = None
        self._segments: Optional[List[torch.Tensor]] = None
        self._predicted_classes: Optional[List[str]] = None
        self._zero_shot_labels: Optional[List[str]] = None
        self._image_processor: Optional[AutoImageProcessor] = None
        self._class_model: Optional[AutoModelForImageClassification] = None
        self._classifier_confidences: Optional[List[float]] = None

    def _ensure_sam_checkpoint(self) -> None:
        if not os.path.exists(ModelConstants.SAM_CHECKPOINT_FILENAME):
            urllib.request.urlretrieve(
                ModelConstants.SAM_CHECKPOINT_URL,
                ModelConstants.SAM_CHECKPOINT_FILENAME,
            )

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
                    # Fallbacks for older/newer Pillow versions without static attributes in stubs
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

    @staticmethod
    def _get_mask_box(tensor: torch.Tensor) -> Tuple[Optional[int], Optional[int]]:
        non_zero_indices = torch.nonzero(tensor, as_tuple=True)[0]
        if non_zero_indices.numel() == 0:
            return None, None
        first_n = non_zero_indices[:1].item()
        last_n = non_zero_indices[-1:].item()
        return first_n, last_n

    def _init_sam(self) -> None:
        self._ensure_sam_checkpoint()
        self._sam_model = sam_model_registry[ModelConstants.SAM_VARIANT](
            checkpoint=ModelConstants.SAM_CHECKPOINT_FILENAME
        )
        self._sam_model.to(self.device)
        self._mask_generator = SamAutomaticMaskGenerator(
            model=self._sam_model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

    def generate_masks(self) -> List[Dict[str, Any]]:
        if self._mask_generator is None:
            self._init_sam()
        image_np = np.array(self._load_image())
        masks = self._mask_generator.generate(image_np)
        self._masks_sorted = sorted(masks, key=lambda x: x["area"], reverse=True)
        return self._masks_sorted

    def build_panoptic_map(self) -> torch.Tensor:
        if self._masks_sorted is None:
            self.generate_masks()
        width, height = self._load_image().size
        panoptic_map_np = np.zeros((height, width), dtype=np.int32)
        for idx, mask_data in enumerate(self._masks_sorted[: self.top_n]):
            panoptic_map_np[mask_data["segmentation"]] = idx + 1
        self._panoptic_map = torch.from_numpy(panoptic_map_np)
        return self._panoptic_map

    def save_panoptic_map_image(self, output_path: str = "pipeline/outputs/panoptic.png", labels: Optional[List[str]] = None, annotate: bool = True) -> str:
        """
        Save the panoptic map as a colorized PNG image using the tab20 colormap.
        If labels are provided (in order of label ids 1..N) and annotate=True,
        overlay the label id and text on each segment.
        Returns the saved file path.
        """
        if self._panoptic_map is None:
            self.build_panoptic_map()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        assert self._panoptic_map is not None
        panoptic_np = self._panoptic_map.cpu().numpy().astype(np.int32)
        max_label = int(panoptic_np.max()) if panoptic_np.size > 0 else 0
        if max_label == 0:
            color_img = np.zeros((*panoptic_np.shape, 3), dtype=np.uint8)
        else:
            cmap = plt.get_cmap("tab20", max(20, max_label + 1))
            # Map each label to a color; label 0 stays black
            colors = (cmap(np.arange(max_label + 1))
                      [:, :3] * 255).astype(np.uint8)
            color_img = colors[panoptic_np]
            color_img[panoptic_np == 0] = 0
        img_pil = Image.fromarray(color_img, mode="RGB")

        if annotate:
            # Ensure we have labels aligned to segment label indices (1..N)
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            # Iterate over present labels to place text at mask centroid
            for label_id in self._iterate_labels():
                # _iterate_labels excludes 0
                mask = self._panoptic_map == label_id
                mask_np = mask.cpu().numpy()
                ys, xs = np.where(mask_np)
                if ys.size == 0:
                    continue
                cx = int(xs.mean())
                cy = int(ys.mean())
                text = str(label_id)
                if labels and 0 <= (label_id - 1) < len(labels):
                    text = f"{label_id}: {labels[label_id - 1]}"
                # Draw white text with black stroke for contrast
                try:
                    draw.text((cx, cy), text, fill=(255, 255, 255), font=font,
                              stroke_width=2, stroke_fill=(0, 0, 0), anchor="mm")
                except TypeError:
                    # Fallback: center manually without anchor support
                    try:
                        bbox = draw.textbbox((0, 0), text, font=font)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                    except Exception:
                        # Rough estimate if textbbox unavailable
                        tw, th = (8 * len(text), 12)
                    tx = int(cx - tw / 2)
                    ty = int(cy - th / 2)
                    # Optional background for readability
                    bg_pad = 2
                    try:
                        draw.rectangle(
                            [tx - bg_pad, ty - bg_pad, tx + tw + bg_pad, ty + th + bg_pad], fill=(0, 0, 0))
                    except Exception:
                        pass
                    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

        img_pil.save(output_path)
        return output_path

    def _iterate_labels(self) -> List[int]:
        if self._panoptic_map is None:
            self.build_panoptic_map()
        labels = self._panoptic_map.unique().tolist()
        return [int(l) for l in labels if int(l) != 0]

    def crop_segments(self) -> List[torch.Tensor]:
        if self._panoptic_map is None:
            self.build_panoptic_map()
        img_tensor = self._to_image_tensor()
        segments: List[torch.Tensor] = []

        for label in self._iterate_labels():
            mask = self._panoptic_map == label
            y_start, y_end = self._get_mask_box(mask)
            x_start, x_end = self._get_mask_box(mask.T)
            if None in (y_start, y_end, x_start, x_end):
                continue

            cropped_tensor = img_tensor[:, y_start : y_end + 1, x_start : x_end + 1]
            cropped_mask = mask[y_start : y_end + 1, x_start : x_end + 1]
            segment = cropped_tensor * cropped_mask.unsqueeze(0)
            segment[:, ~cropped_mask] = self.background_fill
            segments.append(segment)

        self._segments = segments
        return segments

    def _init_classifier(self) -> None:
        if self._image_processor is None or self._class_model is None:
            self._image_processor = AutoImageProcessor.from_pretrained(
                ModelConstants.IMAGE_MODEL_ID
            )
            self._class_model = AutoModelForImageClassification.from_pretrained(
                ModelConstants.IMAGE_MODEL_ID
            ).to(self.device)

    def classify_segments(self) -> List[str]:
        if self._segments is None:
            self.crop_segments()
        self._init_classifier()
        predicted_classes: List[str] = []
        classifier_confidences: List[float] = []
        assert self._segments is not None
        assert self._image_processor is not None
        assert self._class_model is not None

        self._class_model.eval()
        with torch.no_grad():
            for segment in self._segments:
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
                # Softmax max probability as confidence
                try:
                    probs = torch.softmax(logits, dim=-1)
                    conf = float(probs.max(dim=-1).values.item())
                    classifier_confidences.append(conf)
                except Exception:
                    classifier_confidences.append(float("nan"))

        self._predicted_classes = predicted_classes
        self._classifier_confidences = classifier_confidences
        return predicted_classes

    def zero_shot_labels(self) -> List[str]:
        if self._predicted_classes is None:
            self.classify_segments()
        device_index = 0 if (self.device == "cuda" and torch.cuda.is_available()) else -1
        zshot = hf_pipeline(
            "zero-shot-classification",
            model=ModelConstants.ZERO_SHOT_MODEL_ID,
            device=device_index,
        )
        labels: List[str] = []
        assert self._predicted_classes is not None
        for predicted_class in self._predicted_classes:
            result = zshot(
                predicted_class, candidate_labels=self.candidate_labels
            )
            labels.append(result["labels"][0])
        self._zero_shot_labels = labels
        return labels

    def count_candidate_labels(self) -> Dict[str, int]:
        if not self.candidate_labels:
            return {}
        if self._zero_shot_labels is None:
            self.zero_shot_labels()
        assert self._zero_shot_labels is not None
        return {
            label: sum(1 for l in self._zero_shot_labels if l == label)
            for label in self.candidate_labels
        }

    def correct_predictions(
        self,
        classes: Optional[Dict[int, str]] = None,
        labels: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        if self._segments is None:
            raise ValueError("No segments available. Run the pipeline first.")

        if self._predicted_classes is None:
            self.classify_segments()
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

    def plot_original_and_panoptic(self) -> None:
        if self._panoptic_map is None:
            self.build_panoptic_map()
        assert self._panoptic_map is not None
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(self._load_image())
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(self._panoptic_map, cmap="tab20", interpolation="nearest")
        num_segments = len(self._iterate_labels())
        plt.title(f"SAM Segmentation ({num_segments} segments)")
        plt.axis("off")
        plt.show()

    def plot_segments(self) -> None:
        if self._segments is None:
            self.crop_segments()
        assert self._segments is not None
        titles: List[str] = []
        if self._zero_shot_labels is not None and self._predicted_classes is not None:
            for cls, lab in zip(self._predicted_classes, self._zero_shot_labels):
                titles.append(f"Predicted: {cls}, Label: {lab}")
        elif self._predicted_classes is not None:
            titles = [f"Predicted: {c}" for c in self._predicted_classes]
        else:
            titles = ["Segment"] * len(self._segments)

        for segment, title in zip(self._segments, titles):
            plt.imshow(segment.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            plt.title(title)
            plt.axis("off")
            plt.show()

    def run(self, do_zero_shot: bool = True, visualize: bool = False) -> Dict[str, Any]:
        overall_t0 = time.perf_counter()

        # SAM: mask generation (+ basic postprocessing)
        t0 = time.perf_counter()
        self.generate_masks()
        self.build_panoptic_map()
        self.crop_segments()
        sam_ms = (time.perf_counter() - t0) * 1000.0

        # Classifier
        t0 = time.perf_counter()
        self.classify_segments()
        classifier_ms = (time.perf_counter() - t0) * 1000.0

        # Zero-shot (optional)
        zero_shot_ms = 0.0
        if do_zero_shot:
            t0 = time.perf_counter()
            self.zero_shot_labels()
            zero_shot_ms = (time.perf_counter() - t0) * 1000.0

        if visualize or self.show_plots:
            self.plot_original_and_panoptic()
            self.plot_segments()

        run_id = uuid.uuid4().hex
        panoptic_path = os.path.join("pipeline", "outputs", f"{run_id}.png")
        self.save_panoptic_map_image(
            panoptic_path, labels=self._zero_shot_labels, annotate=True)

        # Build models metadata
        models_used: Dict[str, Any] = {
            "sam": {
                "variant": ModelConstants.SAM_VARIANT,
                "checkpoint": ModelConstants.SAM_CHECKPOINT_FILENAME,
            },
            "classifier": {"id": ModelConstants.IMAGE_MODEL_ID},
            "zero_shot": {"id": ModelConstants.ZERO_SHOT_MODEL_ID},
        }

        # Collect run metrics
        metrics_collector = MetricsCollector()
        metadata = metrics_collector.build(
            image=self._original_image,
            segments=self._segments,
            zero_shot_labels=self._zero_shot_labels,
            predicted_classes=self._predicted_classes,
            models_used=models_used,
            classifier_confidences=self._classifier_confidences,
            timings_ms={
                "sam_ms": sam_ms,
                "classifier_ms": classifier_ms,
                "zero_shot_ms": zero_shot_ms,
                "overall_ms": (time.perf_counter() - overall_t0) * 1000.0,
            },
        )
        result: Dict[str, Any] = {
            # "image": self._image,
            # "panoptic_map": self._panoptic_map,
            # "segments": self._segments,
            "predicted_classes": self._predicted_classes,
            "zero_shot_labels": self._zero_shot_labels,
            "label_counts": self.count_candidate_labels(),
            "id": run_id,
            "panoptic_path": panoptic_path,
            "metadata": metadata,
        }
        return result


if __name__ == "__main__":
    pipeline_runner = SamSegmentationClassifier()
    pipeline_runner.image_path = "pipeline/inputs/image2.jpg"
    print(pipeline_runner.run())
