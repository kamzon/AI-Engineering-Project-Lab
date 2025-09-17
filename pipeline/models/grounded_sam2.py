from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from pipeline.config import ModelConstants
from pipeline.utils.image_preprocessor import ImagePreprocessor


class GroundedSAM2:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor: Optional[AutoProcessor] = None
        self._detector: Optional[AutoModelForZeroShotObjectDetection] = None
        self._load_source: Optional[str] = None
        self._pre = ImagePreprocessor(target_size=256)

    def _init(self) -> None:
        if self._processor is None or self._detector is None:
            load_source = getattr(
                ModelConstants,
                "GROUNDING_DINO_MODEL_ID",
                "IDEA-Research/grounding-dino-base",
            )
            self._load_source = load_source
            print(f"Loading GroundingDINO from {load_source}")
            # trust_remote_code is required for GroundingDINO custom heads/utilities
            self._processor = AutoProcessor.from_pretrained(
                load_source, trust_remote_code=True
            )
            self._detector = (
                AutoModelForZeroShotObjectDetection.from_pretrained(
                    load_source, trust_remote_code=True
                ).to(self.device)
            )

    def _to_pil(self, image: Union[Image.Image, torch.Tensor]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, torch.Tensor):
            # Expect CHW tensor in [0,255] or [0,1]
            tensor = image.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                chw = tensor
            elif tensor.ndim == 3 and tensor.shape[2] in (1, 3):
                chw = tensor.permute(2, 0, 1)
            else:
                raise ValueError(
                    "Unsupported tensor shape for image; expected CHW or HWC with 1 or 3 channels"
                )
            if chw.max() <= 1.0:
                chw = (chw * 255.0).round()
            chw = chw.clamp(0, 255).byte()
            np_img = chw.permute(1, 2, 0).numpy()
            if np_img.shape[2] == 1:
                return Image.fromarray(np_img[:, :, 0], mode="L").convert("RGB")
            return Image.fromarray(np_img, mode="RGB")
        raise TypeError("image must be a PIL.Image.Image or torch.Tensor")

    def detect(
        self,
        image: Union[Image.Image, torch.Tensor],
        text_queries: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Runs zero-shot object detection using GroundingDINO.

        Returns a list of detections with keys: 'bbox' (xyxy), 'label', 'score'.
        """
        self._init()
        assert self._processor is not None
        assert self._detector is not None
        if self._load_source is not None:
            print(f"Running detection using model from {self._load_source}")

        pil_image_orig = self._to_pil(image)
        orig_w, orig_h = pil_image_orig.size
        pil_image = self._pre.to_rgb_and_resize(pil_image_orig)

        # Build canonical mapping for label normalization
        def canon(s: str) -> str:
            return s.strip().rstrip(".").lower()
        normalized_queries = [q.strip() for q in text_queries if q and q.strip()]
        query_text = " ".join([q if q.endswith(".") else f"{q}." for q in normalized_queries])
        canon_to_query = {canon(q): q for q in normalized_queries}
        print(f"[GroundedSAM2] text_queries={normalized_queries}")

        inputs = self._processor(images=pil_image, text=query_text, return_tensors="pt")
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        self._detector.eval()
        with torch.no_grad():
            outputs = self._detector(**inputs)

        detections: List[Dict[str, Any]] = []

        # Post-process using HF processor (different versions expose different helpers)
        target_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)
        post_processed: Optional[List[Dict[str, Any]]] = None
        try:
            if hasattr(self._processor, "post_process_grounded_object_detection"):
                post_processed = self._processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )
            elif hasattr(self._processor, "post_process_object_detection"):
                # Fallback: generic object detection (no text threshold)
                post_processed = self._processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=box_threshold,
                    target_sizes=target_sizes,
                )
            else:
                raise AttributeError("No suitable post-process method on processor")
        except Exception as e:
            raise RuntimeError(f"Failed to post-process GroundingDINO outputs: {e}")

        if not post_processed:
            print("[GroundedSAM2] post_processed is empty")
            return detections

        result = post_processed[0]
        # Keys can include: 'boxes', 'scores', 'labels', 'text_labels', 'phrases'
        boxes = result.get("boxes")
        scores = result.get("scores")
        text_labels = result.get("text_labels")
        phrases = result.get("phrases")
        labels = result.get("labels")

        # Move tensors to CPU numpy for easy handling
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu()

        # Prefer text_labels when available per HF deprecation notice
        readable: List[str] = []
        if isinstance(text_labels, list) and text_labels:
            readable = [str(p) for p in text_labels]
        elif isinstance(phrases, list) and phrases:
            readable = [str(p) for p in phrases]
        elif isinstance(labels, list) and labels:
            readable = [str(l) for l in labels]
        else:
            readable = ["object"] * (len(boxes) if boxes is not None else 0)
        print(f"[GroundedSAM2] raw_labels={readable}")

        if boxes is not None and scores is not None:
            for i in range(len(boxes)):
                xyxy = boxes[i].tolist()
                score = float(scores[i]) if i < len(scores) else float("nan")
                raw_label = readable[i] if i < len(readable) else "object"
                canonical = canon_to_query.get(canon(raw_label), raw_label)
                detections.append({
                    "bbox": [float(x) for x in xyxy],
                    "label": canonical,
                    "score": score,
                })
        print(f"[GroundedSAM2] detections={len(detections)} by_label={ {d['label']: sum(1 for x in detections if x['label']==d['label']) for d in detections} }")

        return detections
