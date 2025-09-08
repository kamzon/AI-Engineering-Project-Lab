from typing import Any, Dict, List, Optional


class MetricsCollector:
    """
    Collects and aggregates pipeline metrics into a dictionary.

    Expected inputs:
    - image: PIL.Image.Image (or object with .size -> (width, height))
    - segments: list of tensors/arrays shaped (C, H, W)
    - zero_shot_labels: list of strings produced by the zero-shot step
    - predicted_classes: list of strings from the image classification step
    - models_used: dictionary describing models used (ids/variants)
    """

    def build(
        self,
        *,
        image: Optional[Any],
        segments: Optional[List[Any]],
        zero_shot_labels: Optional[List[str]],
        predicted_classes: Optional[List[str]],
        models_used: Optional[Dict[str, Any]] = None,
        classifier_confidences: Optional[List[float]] = None,
        timings_ms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        # Image resolution
        width: Optional[int] = None
        height: Optional[int] = None
        if image is not None and hasattr(image, "size"):
            size_value = getattr(image, "size", None)
            if isinstance(size_value, (tuple, list)) and len(size_value) == 2:
                w_val, h_val = size_value
                try:
                    width = int(w_val) if w_val is not None else None
                    height = int(h_val) if h_val is not None else None
                except (TypeError, ValueError):
                    width, height = None, None

        # Segments and sizes
        num_segments: int = len(segments) if segments is not None else 0

        avg_segment_area: Optional[float] = None
        if segments:
            areas: List[int] = []
            for seg in segments:
                # Expecting seg shape (C, H, W)
                shape = getattr(seg, "shape", None)
                h: Optional[int] = None
                w: Optional[int] = None
                if isinstance(shape, (tuple, list)) and len(shape) >= 3:
                    # Assume (C, H, W)
                    try:
                        h = int(shape[-2])
                        w = int(shape[-1])
                    except (TypeError, ValueError):
                        h, w = None, None
                if h is None or w is None:
                    h = getattr(seg, "height", None)
                    w = getattr(seg, "width", None)
                    try:
                        h = int(h) if h is not None else None
                        w = int(w) if w is not None else None
                    except (TypeError, ValueError):
                        h, w = None, None
                if h is not None and w is not None:
                    areas.append(h * w)
            if areas:
                avg_segment_area = float(sum(areas)) / float(len(areas))

        # Labels and object types
        label_source: List[str] = zero_shot_labels or predicted_classes or []
        num_object_types: int = len(set(label_source)) if label_source else 0

        # Type of object counted: pick the most frequent label if available
        type_of_object_counted: Optional[str] = None
        if label_source:
            counts: Dict[str, int] = {}
            for label in label_source:
                counts[label] = counts.get(label, 0) + 1
            # Deterministic selection: sort by (-count, label)
            type_of_object_counted = sorted(
                counts.items(), key=lambda kv: (-kv[1], kv[0])
            )[0][0]

        # Count of objects determined by the application
        # Interpreted as the total number of detected objects (segments)
        count_of_objects: int = num_segments

        # Model confidence metrics
        min_classifier_conf: Optional[float] = None
        if classifier_confidences:
            try:
                min_classifier_conf = float(min(classifier_confidences))
            except (TypeError, ValueError):
                min_classifier_conf = None

        # Timing metrics
        timings_ms = timings_ms or {}

        return {
            "image_resolution": {
                "width": width,
                "height": height,
            },
            "type_of_object_counted": type_of_object_counted,
            "count_of_objects": count_of_objects,
            "num_segments": num_segments,
            "num_object_types": num_object_types,
            "avg_segment_resolution": avg_segment_area,
            "models_used": models_used or {},
            # Added metrics
            "classifier_confidences": classifier_confidences or [],
            "classifier_min_confidence": min_classifier_conf,
            "inference_time_ms_per_model": {
                k: float(v) for k, v in timings_ms.items()
            },
            "overall_response_ms": float(timings_ms.get("overall_ms", 0.0)) if "overall_ms" in timings_ms else None,
        }


